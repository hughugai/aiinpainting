import {mkdir,copyFile} from 'node:fs/promises';
const files=['ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm'];
await mkdir('public/onnx-runtime',{recursive:true});
for(const file of files)await copyFile('node_modules/onnxruntime-web/dist/'+file,'public/onnx-runtime/'+file);
