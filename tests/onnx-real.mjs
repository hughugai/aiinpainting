/** Optional: node tests/onnx-real.mjs method /absolute/model.onnx
 * Tests actual WASM graph inference outside the UI; weights are not bundled.
 */
import * as ort from 'onnxruntime-web/wasm';
import {createOnnxInpainter} from '../algorithms/js/onnx.js';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
const [method,path]=process.argv.slice(2);
if(!['migan','lama'].includes(method)||!path)throw Error('Usage: onnx-real.mjs migan|lama /absolute/model.onnx');
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths=resolve('node_modules/onnxruntime-web/dist')+'/';
const w=64,h=48,rgb=new Uint8Array(w*h*3),mask=new Uint8Array(w*h);
for(let y=0;y<h;y++)for(let x=0;x<w;x++){const i=y*w+x;rgb.set([x*3,y*4,120],i*3);if(x>27&&x<36&&y>20&&y<28)mask[i]=1;}
const start=performance.now(),runner=await createOnnxInpainter(ort,method,new Uint8Array(await readFile(path)));
try{
  const result=await runner.run(rgb,mask,w,h);
  for(let i=0;i<w*h;i++)if(!mask[i])for(let c=0;c<3;c++)if(result[i*3+c]!==rgb[i*3+c])throw Error('Known pixel modified');
  if(!result.some((v,i)=>mask[Math.floor(i/3)]&&v!==rgb[i]))throw Error('Model did not fill holes');
  await writeFile('/private/tmp/inpainting-'+method+'-wasm.rgb',result);
  console.log(method,'WASM real inference PASS',Math.round(performance.now()-start)+'ms');
}finally{await runner.dispose();}
