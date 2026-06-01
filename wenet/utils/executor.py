# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import datetime
import logging
import sys
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from wenet.utils.common import StepTimer

from wenet.utils.train_utils import (wenet_join, batch_forward, batch_backward,
                                     update_parameter_and_lr, log_per_step,
                                     save_model)


class Executor:

    def __init__(self,
                 global_step: int = 0,
                 device: torch.device = torch.device("cpu")):
        self.step = global_step + 1
        self.train_step_timer = None
        self.cv_step_timer = None
        self.device = device

        #[AI修改 开始位置 20260206 训练过程统计已消费样本数] 作用：在step保存checkpoint时记录累计训练了多少条数据
        self._num_seen_utts_local = 0
        self._num_seen_tokens_local = 0
        self._num_seen_frames_local = 0
        #[AI修改 结束位置 20260206 训练过程统计已消费样本数]

    #[AI修改 开始位置 20260206 训练过程统计已消费样本数] 作用：汇总各rank累计消费量，写入checkpoint的info_dict
    def _maybe_update_seen_stats(self, batch_dict: dict):
        if batch_dict is None:
            return
        if 'target_lengths' not in batch_dict:
            return
        num_utts = int(batch_dict['target_lengths'].size(0))
        if num_utts <= 0:
            return
        self._num_seen_utts_local += num_utts
        if 'target_lengths' in batch_dict:
            self._num_seen_tokens_local += int(batch_dict['target_lengths'].sum().item())
        if 'feats_lengths' in batch_dict:
            self._num_seen_frames_local += int(batch_dict['feats_lengths'].sum().item())

    def _collect_seen_stats_global(self):
        try:
            import torch.distributed as dist
        except Exception:
            return {
                'num_seen_utts_local': int(self._num_seen_utts_local),
                'num_seen_tokens_local': int(self._num_seen_tokens_local),
                'num_seen_frames_local': int(self._num_seen_frames_local),
            }
        if not dist.is_available() or not dist.is_initialized():
            return {
                'num_seen_utts_local': int(self._num_seen_utts_local),
                'num_seen_tokens_local': int(self._num_seen_tokens_local),
                'num_seen_frames_local': int(self._num_seen_frames_local),
            }

        dev = self.device
        t = torch.tensor([
            int(self._num_seen_utts_local),
            int(self._num_seen_tokens_local),
            int(self._num_seen_frames_local),
        ], dtype=torch.long, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return {
            'num_seen_utts_local': int(self._num_seen_utts_local),
            'num_seen_tokens_local': int(self._num_seen_tokens_local),
            'num_seen_frames_local': int(self._num_seen_frames_local),
            'num_seen_utts_global': int(t[0].item()),
            'num_seen_tokens_global': int(t[1].item()),
            'num_seen_frames_global': int(t[2].item()),
        }
    #[AI修改 结束位置 20260206 训练过程统计已消费样本数]

    def train(self, model, optimizer, scheduler, train_data_loader,
              cv_data_loader, writer, configs, scaler, group_join):
        ''' Train one epoch
        '''
        if self.train_step_timer is None:
            self.train_step_timer = StepTimer(self.step)
        model.train()
        info_dict = copy.deepcopy(configs)
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx
                if wenet_join(group_join, info_dict):
                    break

                if batch_dict["target_lengths"].size(0) == 0:
                    continue

                #[AI修改 开始位置 20260206 训练过程统计已消费样本数] 作用：累计统计本rank已消费的样本/token/帧
                self._maybe_update_seen_stats(batch_dict)
                #[AI修改 结束位置 20260206 训练过程统计已消费样本数]

                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict.get("train_engine", "torch_ddp") in [
                        "torch_ddp", "torch_fsdp"
                ] and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    info_dict = batch_forward(model, batch_dict, scaler,
                                              info_dict, self.device)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer,
                                                    scheduler, scaler,
                                                    info_dict)
                # write training: tensorboard && log
                log_per_step(writer, info_dict, timer=self.train_step_timer)
                save_interval = info_dict.get('save_interval', sys.maxsize)
                if (self.step +
                        1) % save_interval == 0 and self.step != 0 and (
                            batch_idx + 1) % info_dict["accum_grad"] == 0:
                    import torch.distributed as dist
                    # Ensure all ranks start CV at the same time in step mode
                    dist.barrier()

                    #[AI修改 开始位置 20260206 训练过程统计已消费样本数] 作用：在保存step checkpoint前写入累计训练量
                    info_dict.update(self._collect_seen_stats_global())
                    #[AI修改 结束位置 20260206 训练过程统计已消费样本数]

                    loss_dict = self.cv(model, cv_data_loader, configs)
                    model.train()
                    info_dict.update({
                        "tag":
                        "step_{}".format(self.step),
                        "loss_dict":
                        loss_dict,
                        "save_time":
                        datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                        "lrs":
                        [group['lr'] for group in optimizer.param_groups]
                    })
                    save_model(model, info_dict)
                    # write final cv: tensorboard
                    log_per_step(writer, info_dict)
                    # Ensure all ranks start Train at the same time in step mode
                    dist.barrier()
                self.step += 1 if (batch_idx +
                                   1) % info_dict["accum_grad"] == 0 else 0

    def cv(self, model, cv_data_loader, configs):
        ''' Cross validation on
        '''
        if self.cv_step_timer is None:
            self.cv_step_timer = StepTimer(0.0)
        else:
            self.cv_step_timer.last_iteration = 0.0
        model.eval()
        info_dict = copy.deepcopy(configs)
        num_seen_utts, loss_dict, total_acc = 1, {}, []  # avoid division by 0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(cv_data_loader):
                info_dict["tag"] = "CV"
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx
                info_dict["cv_step"] = batch_idx

                num_utts = batch_dict["target_lengths"].size(0)
                if num_utts == 0:
                    continue

                info_dict = batch_forward(model, batch_dict, None, info_dict,
                                          self.device)
                _dict = info_dict["loss_dict"]

                num_seen_utts += num_utts
                total_acc.append(_dict['th_accuracy'].item(
                ) if _dict.get('th_accuracy', None) is not None else 0.0)
                for loss_name, loss_value in _dict.items():
                    if loss_value is not None and "loss" in loss_name \
                            and torch.isfinite(loss_value):
                        loss_value = loss_value.item()
                        loss_dict[loss_name] = loss_dict.get(loss_name, 0) + \
                            loss_value * num_utts
                # write cv: log
                log_per_step(writer=None,
                             info_dict=info_dict,
                             timer=self.cv_step_timer)
        for loss_name, loss_value in loss_dict.items():
            loss_dict[loss_name] = loss_dict[loss_name] / num_seen_utts
        loss_dict["acc"] = sum(total_acc) / len(total_acc)
        return loss_dict
