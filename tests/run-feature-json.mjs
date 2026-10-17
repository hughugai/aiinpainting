import fs from 'node:fs';
import {inpaint} from '../algorithms/js/inpaint.js';
const a=JSON.parse(fs.readFileSync(0,'utf8'));
const result=inpaint(a.rgb,a.mask,a.width,a.height,a.options);
process.stdout.write(JSON.stringify({image:Array.from(result.image),trace:result.trace}));
