import torchvision.models as models
import torch
import timm
import io
import os
from ptflops import get_model_complexity_info

# GMACs = 0.5 * GFLOPs

with torch.cuda.device(0):
    # net = models.mnasnet1_0() # 0.33 GMac, 4.38 M
    # net = models.mnasnet1_3() # 0.55 GMac, 6.28 M
    # net = models.efficientnet_b0() # 0.4 GMac, 5.29 M
    # net = models.mobilenet_v2() # 0.32 GMac, 3.5 M
    # net = models.mobilenet_v3_large() # 0.23 GMac, 5.48 M
    # net = timm.create_model('tf_efficientnet_b0_ns') # 0.38 GMac, 5.29 M
    # net = timm.create_model('efficientnet_b1_pruned') # 0.31 GMac, 6.33 M
    # net = timm.create_model('mixnet_s') # 0.24 GMac, 4.13 M
    # net = timm.create_model('mixnet_m') # 0.34 GMac, 5.01 M
    # net = timm.create_model('mnasnet_b1') # 0.32 GMac, 4.38 M, 74.658 - 92.114
    # net = timm.create_model('mnasnet_a1') # 0.32 GMac, 3.89 M, 75.448 - 92.604
    # net = timm.create_model('mobilenetv2_140') # 0.59 GMac, 6.11 M
    # net = timm.create_model('mobilenetv2_120d') # 0.68 GMac, 5.8 M
    net = timm.create_model('nasnetalarge') # 
    print(net)
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # input_names = ['input']
    # output_names = ['output']
    # onnx_bytes = io.BytesIO()
    # zero_input = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
    # dynamic_axes = {input_names[0]: {0: 'batch'}}
    # onnx_model = os.path.join("scripts/bmk", "mobilenetv2_120d.onnx")

    # for _, name in enumerate(output_names):
    #     dynamic_axes[name] = dynamic_axes[input_names[0]]
    # extra_args = {'opset_version': 10, 'verbose': False,
    #             'input_names': input_names, 'output_names': output_names,
    #             'dynamic_axes': dynamic_axes}
    # torch.onnx.export(net, zero_input, onnx_bytes, **extra_args)
    # with open(onnx_model, 'wb') as out:
    #     out.write(onnx_bytes.getvalue())

    # print(f"[ONNX] Exported at {onnx_model}")