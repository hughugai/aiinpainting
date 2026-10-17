"""ONNX Runtime inference matching js/onnx.js. No training or implicit downloads."""
import numpy as np
from classical import validate


def pack_inputs(method,rgb,mask):
    rgb,hole=validate(rgb,mask)
    if method not in ('migan','lama'):
        raise ValueError('Unknown ONNX contract')
    rgb[hole]=0
    if method=='migan':
        image=rgb.transpose(2,0,1)[None].copy()
        m=((~hole).astype(np.uint8)*255)[None,None]
    else:
        image=(rgb.astype(np.float32)/255).transpose(2,0,1)[None].copy()
        m=hole.astype(np.float32)[None,None]
    return {'image':image,'mask':m}


def inpaint_onnx(method,model,rgb,mask,image_name='image',mask_name='mask',output_name=None,output_scale=1,fixed_size=512):
    import onnxruntime as ort
    original,hole=validate(rgb,mask)
    if not hole.any():
        return original
    session=ort.InferenceSession(str(model),providers=['CPUExecutionProvider'])
    h,w=hole.shape
    if method=='lama' and fixed_size:
        if w>fixed_size or h>fixed_size:
            raise ValueError('Working image exceeds fixed model size')
        rgb=np.pad(rgb,((0,fixed_size-h),(0,fixed_size-w),(0,0)),mode='edge')
        mask=np.pad(hole,((0,fixed_size-h),(0,fixed_size-w)))
    packed=pack_inputs(method,rgb,mask)
    names={v.name for v in session.get_inputs()}
    if image_name not in names or mask_name not in names:
        raise ValueError('Model does not match image/mask contract')
    result=session.run([output_name or session.get_outputs()[0].name],
                       {image_name:packed['image'],mask_name:packed['mask']})[0]
    ph,pw=rgb.shape[:2]
    if result.shape!=(1,3,ph,pw) or not np.isfinite(result).all():
        raise ValueError('Invalid ONNX output shape or values')
    scale=output_scale
    prediction=np.floor(result[0].transpose(1,2,0).astype(np.float64)*scale+.5).clip(0,255).astype(np.uint8)
    original[hole]=prediction[:h,:w][hole]
    return original
