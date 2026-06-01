import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "decode_from_kaldi_dir.py"


def load_module():
    spec = importlib.util.spec_from_file_location("decode_from_kaldi_dir", str(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DecodeFromKaldiDirTest(unittest.TestCase):
    def setUp(self):
        self.work_dir = Path(tempfile.mkdtemp(prefix="decode_from_kaldi_dir_"))

    def tearDown(self):
        shutil.rmtree(str(self.work_dir), ignore_errors=True)

    def test_build_raw_jsonl_from_kaldi_dir(self):
        data_dir = self.work_dir / "data"
        data_dir.mkdir()
        (data_dir / "text").write_text("utt1 hello world\nutt2 ni hao\n", encoding="utf-8")
        (data_dir / "wav.scp").write_text(
            "utt1 /tmp/a.wav\nutt2 /tmp/b.wav\n",
            encoding="utf-8",
        )
        jsonl_path = self.work_dir / "data.jsonl"

        module = load_module()
        rows = module.build_jsonl_records(data_dir)
        module.write_jsonl(rows, jsonl_path)

        lines = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            lines,
            [
                {"key": "utt1", "txt": "hello world", "wav": "/tmp/a.wav"},
                {"key": "utt2", "txt": "ni hao", "wav": "/tmp/b.wav"},
            ],
        )

    def test_build_recognize_command_uses_raw_audio_path(self):
        module = load_module()
        output_dir = self.work_dir / "out"
        jsonl_path = output_dir / "data.jsonl"
        command = module.build_recognize_command(
            python_bin="python",
            recognize_script="wenet/bin/recognize.py",
            config="exp/model/config.yaml",
            checkpoint="exp/model/model.pt",
            test_data=str(jsonl_path),
            result_dir=str(output_dir / "decode"),
            modes=["rnnt_beam_search"],
            data_type="raw",
            batch_size=1,
            beam_size=10,
            device="cpu",
            num_decoding_left_chunks=5,
            decoding_chunk_size=16,
            simulate_streaming=True,
            ctc_weight=0.1,
            override_config=["dataset_conf.use_precomputed_feat=false"],
        )
        command_str = " ".join(command)
        self.assertIn("--data_type raw", command_str)
        self.assertIn("--test_data {0}".format(jsonl_path), command_str)
        self.assertIn("--override_config dataset_conf.use_precomputed_feat=false", command_str)

    def test_write_compare_outputs(self):
        module = load_module()
        refs = {
            "utt1": "hello",
            "utt2": "world",
        }
        hyps = {
            "utt1": "hello",
            "utt2": "word",
        }
        analysis_dir = self.work_dir / "analysis"

        compare_path, summary_path = module.write_analysis(refs, hyps, analysis_dir)

        compare_lines = compare_path.read_text(encoding="utf-8").splitlines()
        summary = summary_path.read_text(encoding="utf-8")

        self.assertEqual(compare_lines[0], "key\tref\thyp\tstatus")
        self.assertIn("utt1\thello\thello\tOK", compare_lines)
        self.assertIn("utt2\tworld\tword\tERR", compare_lines)
        self.assertIn("total\t2", summary)
        self.assertIn("exact_match\t1", summary)

    def test_build_vc_submit_command_from_args(self):
        module = load_module()

        class Args(object):
            image = "docker.v2.aispeech.com/hpc/ai_on_device-hao.li_k2:pytorch2.1.0-cuda11.8-v1.4.0"
            vc_partition = "pdgpu-a10"
            vc_project = "a-i-o"
            vc_job_name = "wenet-decode"
            vc_cpu_per_task = 16
            vc_mem_per_task = "80G"
            vc_gpu_per_task = 1

        command = module.build_vc_submit_command_from_args(
            args=Args(),
            decode_nj=1,
            log_dir=self.work_dir / "log",
            recognize_command=["python", "wenet/bin/recognize.py", "--config", "x"],
        )
        command_str = " ".join(command)
        self.assertIn("-p pdgpu-a10", command_str)
        self.assertIn("-pj a-i-o", command_str)
        self.assertIn("-g 1", command_str)
        self.assertIn("JOB=1:1", command_str)


if __name__ == "__main__":
    unittest.main()
