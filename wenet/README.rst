transfomer
  line:134
    def tie_or_clone_weights(self, jit_mode: bool = True):
        if hasattr(self,"decoder") and self.decoder is not None:
            if hasattr(self,"tie_or_clone_weights"):
                self.decoder.tie_or_clone_weights(jit_mode)

transducer
  line:24
    rnnt: False,

utils
  line:120
    if model_type == "transducer" and configs.get('model_conf',{}).get('rnnt',False):
        decoder = None
    else:
        decoder = WENET_DECODER_CLASSES[decoder_type](vocab_size,
                                                  encoder.output_size(),
                                                  **configs['decoder_conf'])





python -c "
import torch
m = torch.load('exp/conformer_debug/init.pt', map_location='cpu')
print('=== 检查点中的所有键 ===')
for key in m.keys():
    print(f'{key}: {type(m[key])}')
"
