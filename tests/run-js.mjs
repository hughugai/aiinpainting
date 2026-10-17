import {inpaint} from '../algorithms/js/classical.js';
import {packInputs} from '../algorithms/js/onnx.js';
let text='';for await(const chunk of process.stdin)text+=chunk;
const cases=JSON.parse(text);
const results=cases.map(({method,rgb,mask,w,h,options={}})=>{
  try {return {image:Array.from(inpaint(method,Uint8Array.from(rgb),Uint8Array.from(mask),w,h,options))};}
  catch(e){return {error:e.message};}
});
const p=packInputs('migan',new Uint8Array([11,22,33,44,55,66]),new Uint8Array([0,1]),2,1);
if(p.image.join(',')!=='11,0,22,0,33,0'||p.mask.join(',')!=='255,0')throw Error('MI-GAN tensor contract');
process.stdout.write(JSON.stringify(results));
