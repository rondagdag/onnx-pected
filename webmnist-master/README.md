# webmnist

> npm install http-server

> http-server

browser: http://127.0.0.1:8080/index.html






How to train and create model.onnx

> conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

Run train.py

> python train.py

This is how it saves the model to ONNX

    def save_model(model, device, path):
        # create dummy variable to traverse graph
        x = torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255
        onnx.export(model, x, path)
        print('Saved model to {}'.format(path))


Index.html has link to onnx.js

<script src="https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js"></script>


app.js loads the model and predicts

// load model
this.session = new onnx.InferenceSession();
await this.session.loadModel('model.onnx');


// predict
predict: async function() {
    const digit = this.getDigit();
    const d = [new Tensor(new Float32Array(digit), 'float32', [1,784])];
    
    const output = await this.session.run(d);
    
    this.probabilities = output.values().next().value.data;
    this.prediction = this.probabilities.indexOf(Math.max(...this.probabilities));
}




source
https://github.com/sethjuarez/webmnist