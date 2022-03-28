/*jshint esversion: 9 */
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.8.0/dist/tf.min.js");
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface@0.0.7/dist/blazeface.min.js");
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@3.8.0/dist/tf-backend-wasm.min.js");

let initialized = false;
let model;

tf.wasm.setWasmPaths({
    'tfjs-backend-wasm.wasm': 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@3.8.0/wasm-out/tfjs-backend-wasm.wasm',
    'tfjs-backend-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@3.8.0/wasm-out/tfjs-backend-wasm-simd.wasm',
    'tfjs-backend-wasm-threaded-simd.wasm': 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@3.8.0/wasm-out/tfjs-backend-wasm-threaded-simd.wasm'
});
tf.setBackend('wasm').then(
    async ()=>{
        model = await blazeface.load();
        initialized=true;
    }
);


onmessage = function(img) {
    //create ImageData object for use in tfjs
    const image = new ImageData(img.data.data, img.data.width, img.data.height);

    // tf.wasm.setWasmPaths({
    //     'tfjs-backend-wasm.wasm': 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@3.8.0/wasm-out/tfjs-backend-wasm.wasm',
    //     'tfjs-backend-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@3.8.0/wasm-out/tfjs-backend-wasm-simd.wasm',
    //     'tfjs-backend-wasm-threaded-simd.wasm': 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@3.8.0/wasm-out/tfjs-backend-wasm-threaded-simd.wasm'
    // });
    // tf.setBackend('wasm').then

    if(initialized){
        detectFace(
            tf.tidy(()=>{
                    return tf.browser.fromPixels(image);
            }),img.data.width,img.data.height
        );
    } else {
        postMessage([[0,0], [1,1]]);
    }


    //console.table( tf.memory() );

};

async function detectFace(img, width, height) {

    tf.engine().startScope();

    const returnTensors = false;
    const predictions = await model.estimateFaces(img, returnTensors);
    var cropArea = null;
    if (predictions.length > 0) {
        /*
        `predictions` is an array of objects describing each detected face, for example:

        [
          {
            topLeft: [232.28, 145.26],
            bottomRight: [449.75, 308.36],
            probability: [0.998],
            landmarks: [
              [295.13, 177.64], // right eye
              [382.32, 175.56], // left eye
              [341.18, 205.03], // nose
              [345.12, 250.61], // mouth
              [252.76, 211.37], // right ear
              [431.20, 204.93] // left ear
            ]
          }
        ]
        */

        //TODO add relative border and adjust ratio to screen size
        var start = predictions[0].topLeft;
        var end = predictions[0].bottomRight;

        cropArea = CalculateCroppingArea(start, end, width, height);
    }
    postMessage(cropArea);

    tf.engine().endScope();
    img.dispose();
}

function CalculateCroppingArea(start, end, width, height){

    var cropWidth = (end[0] - start[0]);
    var cropHeight = (end[1] - start[1]);

    var cropCenter = [start[0] + cropWidth/2, start[1] + cropHeight/2];

    cropWidth *= 1.4;
    cropHeight *= 1.4;

    if(cropWidth > width){
        cropWidth = width;
    }

    if(cropWidth > height){
        cropWidth = height;
    }



    start = [cropCenter[0] - (cropWidth)/2, cropCenter[1] - (cropWidth)/2];
    end = [cropCenter[0] + (cropWidth)/2, cropCenter[1] + (cropWidth)/2];

    //normalize predicted area
    start[0] /= width;
    end[0] /= width;
    start[1] /= height;
    end[1] /= height;

    if(start[0] < 0){
        end[0] += Math.abs(start[0]);
        start[0] = 0;
    }
    if (end[0] > 1){
        start[0] -= end[0]-1;
        end[0] = 1;
    }

    if (end[1] > 1){
        start[1] -= end[1]-1;
        end[1] = 1;
    }
    if(start[1] < 0){
        end[1] += Math.abs(start[1]);
        start[1] = 0;
    }

    //combine area
    var cropArea = [start, end];
    return cropArea;
}