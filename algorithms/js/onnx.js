/** Runtime-injected API: works with onnxruntime-web in websites and WASM workers.
 * MI-GAN: official migan_pipeline_v2.onnx, uint8 NCHW, mask 255=known.
 * LaMa: export contract image/mask float32 NCHW [0,1], mask 1=hole,
 * output NCHW RGB [0,255] (Carve export). Other exports need explicit options.
 */
import {validate} from './classical.js';
export const LAMA_URL='https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx';
export const MIGAN_URL='https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx';
export function packInputs(method,rgb,mask,w,h) {
  validate(rgb,mask,w,h);
  if(!['migan','lama'].includes(method))throw Error('Unknown ONNX contract');
  const n=w*h,bytes=method==='migan',image=bytes?new Uint8Array(n*3):new Float32Array(n*3),m=bytes?new Uint8Array(n):new Float32Array(n);
  for(let i=0;i<n;i++){m[i]=bytes?(mask[i]?0:255):(mask[i]?1:0);for(let c=0;c<3;c++)image[c*n+i]=bytes?(mask[i]?0:rgb[i*3+c]):(mask[i]?0:rgb[i*3+c]/255);}
  return {image,mask:m,type:bytes?'uint8':'float32',imageDims:[1,3,h,w],maskDims:[1,1,h,w]};
}
export function composite(rgb,mask,w,h,tensor,scale=1) {
  if(tensor.dims.join(',')!==[1,3,h,w].join(','))throw Error('Expected NCHW output matching the working image');
  const out=Uint8Array.from(rgb),n=w*h;
  for(let i=0;i<n;i++)if(mask[i])for(let c=0;c<3;c++) {const v=Number(tensor.data[c*n+i])*scale;if(!Number.isFinite(v))throw Error('Non-finite ONNX output');out[i*3+c]=Math.max(0,Math.min(255,Math.floor(v+.5)));}
  return out;
}
export async function createOnnxInpainter(ort,method,model,{executionProviders=['wasm'],imageName='image',maskName='mask',outputName,outputScale=1,fixedSize=method==='lama'?512:0}={}) {
  if(!['migan','lama'].includes(method))throw Error('Unknown ONNX contract');
  const session=await ort.InferenceSession.create(model,{executionProviders});
  if(!session.inputNames.includes(imageName)||!session.inputNames.includes(maskName)) {await session.release();throw Error('Model inputs do not match image/mask contract');}
  const output=outputName??session.outputNames[0];
  if(!session.outputNames.includes(output)){await session.release();throw Error('Missing configured output');}
  return {
    async run(rgb,mask,w,h) {
      validate(rgb,mask,w,h);
      if(!mask.some(Boolean))return Uint8Array.from(rgb);
      const input=fixedSize?padSquare(rgb,mask,w,h,fixedSize):{rgb,mask,w,h};
      const p=packInputs(method,input.rgb,input.mask,input.w,input.h);
      const feeds={[imageName]:new ort.Tensor(p.type,p.image,p.imageDims),[maskName]:new ort.Tensor(p.type,p.mask,p.maskDims)};
      const result=await session.run(feeds);
      const filled=composite(input.rgb,input.mask,input.w,input.h,result[output],outputScale);
      if(!fixedSize)return filled;
      const cropped=Uint8Array.from(rgb);
      for(let y=0;y<h;y++)cropped.set(filled.subarray(y*input.w*3,(y*input.w+w)*3),y*w*3);
      return cropped;
    },
    dispose:()=>session.release(),
  };
}

/** Pad a working image without geometric distortion, then crop predictions back.
 * Fixed-shape LaMa exports commonly require 512x512 even for non-square photos.
 */
export function padSquare(rgb,mask,w,h,size=512) {
  if(w>size||h>size)throw Error('Working image exceeds fixed model size');
  const image=new Uint8Array(size*size*3),hole=new Uint8Array(size*size);
  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
    const p=y*size+x,q=Math.min(y,h-1)*w+Math.min(x,w-1);
    image.set(rgb.subarray(q*3,q*3+3),p*3);
    if(y<h&&x<w)hole[p]=mask[y*w+x];
  }
  return {rgb:image,mask:hole,w:size,h:size};
}
