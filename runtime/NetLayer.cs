using System.Collections;
using System.Collections.Generic;
using Cysharp.Threading.Tasks;// System.Threading.Tasks;

using UnityEngine;
using UnityEngine.Rendering;
namespace EyE.NNET
{
    /// <summary>
    /// Class used by NeuralNetworks to store data about the neurons and connections between them, and the neurons in the previous layer.
    /// in this variant all neurons in the layer use the same activation function.
    /// </summary>
    [System.Serializable]
    public class NetLayer
    {
        [SerializeField]
        protected int numNeurons;
        public int NumNeurons => numNeurons;

        [SerializeField]
        private int numInputs;
        public int NumInputs => numInputs;

        [SerializeField]
        public float[,] weights; // indexes are [neuronIndex,inputIndex].  The weights for each connection are multiplied by the connection's input value, during the ComputeOutputs operation, to generate the connection's weighted input to the neuron.
        [SerializeField]
        public float[] biases; // index by neuronIndex.  The Bias value is added to the sum of all the weighted inputs of a neuron to generate it's unactivated output value.
        [SerializeField]
        public ActivationFunction activationFunction; //this activation function applies to all neurons in the layer- the unactivated output is passed to this function, and the result is the final output of the neuron.


        public float[] inputs; //stores a copy of the last inputs provided.  used as output for display, and during backpropegation)
        protected float[] outputs; // stores a copy of the last outputs generated (used during backpropegation)

        /// <summary>
        /// stores a copy of the last outputs generated (for potential display)
        /// </summary>
        public float[] lastOutputs=>outputs;

        public NetLayer(int numNeurons, int numInputs, ActivationFunction activationFunction)
        {
            this.numNeurons = numNeurons;
            this.numInputs = numInputs;
            this.activationFunction = activationFunction;
            InitializeWeightsAndBiasesRandom0to1();
            outputs = new float[numNeurons];
            gradientBiases = new float[numNeurons];
            gradientWeights = new float[numNeurons, numInputs];
            propagatedErrors = new float[numInputs]; // Errors to be propagated to the previous layer
            sourceErrors = new float[numNeurons];
        }

        public NetLayer(NetLayer source)
        {
            this.numNeurons = source.numNeurons;
            this.numInputs = source.numInputs;
            this.activationFunction = source.activationFunction;
            this.weights = (float[,])source.weights.Clone();
            this.biases = (float[])source.biases.Clone();
            this.activationFunction = source.activationFunction;
            outputs = new float[numNeurons];
            gradientBiases = new float[numNeurons];
            gradientWeights = new float[numNeurons, numInputs];
            propagatedErrors = new float[numInputs]; // Errors to be propagated to the previous layer
            sourceErrors = new float[numNeurons];
        }

        public virtual NetLayer Clone()
        {
            return new NetLayer(this);
        }

        public NetLayer CloneAndMutate(float activationFunctionChangeChance, float biasMutationChance, float biasMutationAmount, float numNeuronsMutationChance, float numNeuronsMutationAmount, float weightsMutationChance, float weightsMutationAmount)
        {
            NetLayer originalLayer = this;
            NetLayer clonedLayer = (NetLayer)this.Clone();

            // Random value between (1- mutationAmount/2) and (1 + mutationAmount/2), so 1.0f is exactly in the middle
            float mutateCoef(float chanceOfMutation, float mutationAmount)
            {
                if (chanceOfMutation > Random.value)
                    return 1.0f;
                return (Random.value * mutationAmount) * 2 + 1;
            }
            int[] GetRandomIndexes(int numberOfIndexesToGet, int totalNumberOfIndexes)
            {
                if (totalNumberOfIndexes <= 0 || numberOfIndexesToGet <= 0 || totalNumberOfIndexes < numberOfIndexesToGet)
                {
                    throw new System.ArgumentException("Invalid input. Ensure that Y > X and both X and Y are positive integers.");
                }


                int[] numbers = new int[totalNumberOfIndexes + 1];

                // Fill the numbers array with values from 0 to Y
                for (int i = 0; i <= totalNumberOfIndexes; i++)
                {
                    numbers[i] = i;
                }

                // Perform Fisher-Yates shuffle to shuffle the numbers
                for (int i = totalNumberOfIndexes; i > 0; i--)
                {
                    int j = Random.Range(0, i + 1);
                    int temp = numbers[i];
                    numbers[i] = numbers[j];
                    numbers[j] = temp;
                }

                // Select the first X elements from the shuffled array
                int[] selectedNumbers = new int[numberOfIndexesToGet];
                System.Array.Copy(numbers, selectedNumbers, numberOfIndexesToGet);

                return selectedNumbers;
            }
            clonedLayer.numNeurons = (int)((float)numNeurons * mutateCoef(numNeuronsMutationChance, numNeuronsMutationAmount));
            float[,] newWeights = new float[clonedLayer.numNeurons, numInputs];
            float[] newBiases = new float[clonedLayer.numNeurons];
            if (clonedLayer.numNeurons >= numNeurons)// all old neurons will be used (and new ones may be created)
            {
                for (int i = 0; i < clonedLayer.numNeurons; i++)
                {
                    for (int j = 0; j < numInputs; j++)
                    {
                        if (i < numNeurons) // is this a "new" neuron
                            newWeights[i, j] = weights[i, j] * mutateCoef(weightsMutationChance, weightsMutationAmount);
                        else
                            newWeights[i, j] = Random.value;// Initialize with random weights
                    }
                    if (i < numNeurons)
                        newBiases[i] = biases[i] * mutateCoef(biasMutationChance, biasMutationAmount);
                    else
                        newBiases[i] = Random.value;// Initialize with random biases
                }
            }
            else //fewer neurons in new layer- which should we keep?
            {
                int[] neuronsToUse = GetRandomIndexes(clonedLayer.numNeurons, numNeurons);
                for (int i = 0; i < neuronsToUse.Length; i++)
                {
                    int neuronIndex = neuronsToUse[i];
                    for (int j = 0; j < numInputs; j++)
                    {
                        newWeights[i, j] = weights[neuronIndex, j] * mutateCoef(weightsMutationChance, weightsMutationAmount);
                    }
                    newBiases[i] = biases[neuronIndex] * mutateCoef(biasMutationChance, biasMutationAmount);
                }
            }
            clonedLayer.weights = newWeights;
            clonedLayer.biases = newBiases;
            if (Random.value < activationFunctionChangeChance)
            {
                clonedLayer.activationFunction = ActivationFunctionExtension.RandomActivationFunction();
            }
            return clonedLayer;
        }

        //invoked during construction of a new net layer, it assigns random values between 0 and 1 to all weights and biases in the layer
        private void InitializeWeightsAndBiasesRandom0to1()
        {
            weights = new float[numNeurons, numInputs];
            biases = new float[numNeurons];
            float xavierScale = Mathf.Sqrt(2.0f / (numInputs + numNeurons));
            for (int i = 0; i < numNeurons; i++)
            {
                for (int j = 0; j < numInputs; j++)
                {
                    weights[i, j] = Random.Range(-xavierScale, xavierScale);// value;// Initialize with random weights
                }
                biases[i] = Random.value;// Initialize with random biases
            }
        }

        /// <summary>
        /// Generate an output value for each neuron in the layer.  The number of inputs must match the number of input connection that each and every neuron in this layer has.  (The expected value is the number of neurons in the previous layer, or, for the first layer, the number of NeuronNet inputs.)
        /// This variant runs directly on the processor, in this function definition, it does not actually await anything, and so, blocks the error about that.
        /// </summary>
        /// <param name="inputs">an array specifying the input values for the layer to process</param>
        /// <returns>the output values process by the layer. These will either be used an inputs to the next layer, or provide the output for the NeuralNetwork.</returns>
        public virtual UniTask<float[]> ComputeOutputs(float[] inputs)
        {
            this.inputs = inputs;
            if (inputs.Length != numInputs)
                throw new System.ArgumentException("Invalid number of inputs passed to Layer.ComputeOutputs."); //only for now while numInputsPerNeuron== previous layer numb neurons


            for (int i = 0; i < numNeurons; i++)
            {
                float activation = biases[i];
                for (int j = 0; j < numInputs; j++)
                {
                    activation += inputs[j] * weights[i, j];
                }
                outputs[i] = activationFunction.Activate(activation);
            }
            //lastOutputs = outputs;
            return new UniTask<float[]>(outputs);// Task.FromResult<float[]>(outputs);
        }

        protected float[] gradientBiases; //Error in the biases of each neuron on the layer
        protected float[,] gradientWeights;// Error in the weight of each connection from each neuron.  [neuronInexInLayer,connectionIndexInNeuron]
        public float[] propagatedErrors;// Errors to be propagated to the previous layer, one for each neuron on the previous layer.
        public float[] sourceErrors;// Record of the last errors in passed into backpropegate function- one for each output/neuron in the layer.

        /// <summary>
        /// Computes the changes to weight and biases in this layer, and returns the errors that should be passed to the previous layer (previous layer because: the neuron net will call this function for the layers in reverse order).
        /// This function used cashed values from the previous think operation, such as lastInputs and lastOutputs.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="errors"></param>
        /// <param name="learningRate"></param>
        /// <returns></returns>
        public virtual UniTask<float[]> Backpropagate(float[] inputs, float[] errors, float learningRate)
        {
            if (inputs.Length != numInputs || errors.Length != numNeurons)
                throw new System.ArgumentException("Invalid number of inputs or errors passed to Layer.Backpropagate.");

            /*Debug.Log("Doing backpropegate for layer of type: " + GetType()
                + "\n layer inputs: " + string.Join(",", inputs)
                + "\n layer outputs: " + string.Join(",", lastOutputs)
                + "\n layer errors: " + string.Join(",", errors));
            */
            sourceErrors = errors;

            for (int j = 0; j < numInputs; j++)
            { propagatedErrors[j] = 0; }

            // Compute gradients for biases and weights
            for (int i = 0; i < numNeurons; i++)
            {
                float lastOutputActivationDerivative = activationFunction.ApplyActivationFunctionDerivative(lastOutputs[i]);
                // Compute the gradient for biases
                gradientBiases[i] = errors[i] * lastOutputActivationDerivative;
                // Update biases
                biases[i] -= learningRate * gradientBiases[i];

                // Compute the gradient for weights
                for (int j = 0; j < numInputs; j++)
                {
                    gradientWeights[i, j] = errors[i] * lastOutputActivationDerivative * inputs[j];

                    // Accumulate errors to be propagated to the previous layer
                    propagatedErrors[j] += errors[i] * lastOutputActivationDerivative * weights[i, j];
                    //update weights
                    weights[i, j] -= learningRate * gradientWeights[i, j];
                }
            }
            return new UniTask<float[]> (propagatedErrors);
        }

    }

    /// <summary>
    /// This Variant of the NetLayer base class overrides the ComputeOutput functions to use a compute shader, rather than process it all on the cpu.
    /// It also has a couple of new functions called ComputeLayer.  These function will NOT automatically pass data back to the cpu after processing.  This allows data to be passed from one layer to the next, while staying on the GPU, which enhances performance.  
    ///     The version that has no parameter will use the output generated by the prious layer, as stored on the GPOU.  
    ///     The other version, which takes inputs, is really only intended for the first layer, to get the external input into the net.
    /// To get the output data back to the CPU use the GetLastComputedOutput function.  While the ComputeOutput functions do this automatically, every cycle, the ComputeLayer versions wont.
    /// Alternatively, you can use the GetGPUData function to pull all the information about the layer back to the cpu.  This can use useful if you want to visualize or display the network, somehow.
    /// </summary>
    public class ComputeShaderLayer : NetLayer, System.IDisposable
    {
        // Reference to the compute shader asset
        public ComputeShader computeShader;
        public bool computeShaderIsSinglePass = true;
         
        //these string are exact copies of function identifiers in the compute shader code
        private static readonly string[] singlePassComputeLayerKernalNames = new string[4] { "ComputeLayerNone", "ComputeLayerReLU", "ComputeLayerSigmoid", "ComputeLayerTanh01" };

        private static readonly string resetZeroOutputKernalName = "ZeroOutputs";
        private static readonly string[] biasAndActivationkernelNames = new string[4] { "ApplyBiasAndActivationNone", "ApplyBiasAndActivationReLU", "ApplyBiasAndActivationSigmoid", "ApplyBiasAndActivationTanh01" };
        private static readonly string computeWeightSumKernalName = "OneDimComputeLayerWeightedSum";
        private static readonly string backPrepegateZeroStartKernalName = "BackPropegateZeroErrorsStart";
        private static readonly string[] backPrepegateKernalNames = new string[4] { "BackPropegateNoneSinglePass", "BackPropegateReLuSinglePass", "BackPropegateSigmoidSinglePass", "BackPropegateTanh01SinglePass", };
        
        private static readonly string[] backPrepegateMultiPassKernalNames = new string[4] { "BackPropegateNoneComputeActivationDerivatiePass", "BackPropegateReLuComputeActivationDerivatiePass", "BackPropegateSigmoidComputeActivationDerivatiePass", "BackPropegateTanh01ComputeActivationDerivatiePass", };
        private static readonly string BackPropegateInputPassName = "BackPropegateInputPass";

        int singlePassComputeLayerKernalIndex;
        int resetZeroOutputKernalIndex;
        int computeWeightSumKernalIndex;
        int computeAddBiasAndActivationKernelIndex;
        int BackPropegateBiasPassKernalIndex;
        int BackPropegateInputPassKernalIndex;
        int backPropZeroKernalIndex;
        int backPropAlterKernalIndex;


        // Buffers for inputs, outputs, weights, and biases
        private ComputeBuffer inputBuffer;
        public ComputeBuffer outputBuffer;
        private ComputeBuffer weightsBuffer;
        private ComputeBuffer biasesBuffer;

        private ComputeBuffer propegatedErrorBuffer;
        public ComputeBuffer sourceErrorBuffer;
        public ComputeBuffer activationDerivativeBuffer;
        //private ComputeBuffer weightsErrorBuffer;
        //private ComputeBuffer biasesErrorBuffer;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="computeShader"></param>
        /// <param name="numNeurons"></param>
        /// <param name="numInputs"></param>
        /// <param name="activationFunction"></param>
        /// <param name="inputBuffer">may contain a referene to the output buffer of the previous layer.  This way the data can stay on the GPU</param>
        public ComputeShaderLayer(ComputeShader computeShader,int numNeurons, int numInputs, ActivationFunction activationFunction, bool computeShaderIsSinglePass = true, ComputeShaderLayer previousLayer =null) :base(numNeurons,  numInputs,  activationFunction)
        {
            this.computeShader = ComputeShader.Instantiate<ComputeShader>(computeShader);
            this.computeShaderIsSinglePass = computeShaderIsSinglePass;
            if (previousLayer != null)
            {
                this.inputBuffer = previousLayer.outputBuffer;
            //    this.propegatedErrorBuffer = previousLayer.sourceErrorBuffer;
            }

            InitBuffers();
        }

        public ComputeShaderLayer(ComputeShader computeShader,NetLayer source, bool computeShaderIsSinglePass = true,ComputeShaderLayer previousLayer = null) : base(source)
        {
            this.computeShader = ComputeShader.Instantiate<ComputeShader>(computeShader);
            this.computeShaderIsSinglePass = computeShaderIsSinglePass;
            if (previousLayer != null)
            {
                this.inputBuffer = previousLayer.outputBuffer;
          //      this.propegatedErrorBuffer = previousLayer.sourceErrorBuffer;
            }
            InitBuffers();
        }


        void InitBuffers()
        {
            if (computeShaderIsSinglePass)
                this.singlePassComputeLayerKernalIndex = computeShader.FindKernel(singlePassComputeLayerKernalNames[(int)activationFunction]);
            else
            {
                this.computeWeightSumKernalIndex = computeShader.FindKernel(computeWeightSumKernalName);
                this.resetZeroOutputKernalIndex = computeShader.FindKernel(resetZeroOutputKernalName);
                this.computeAddBiasAndActivationKernelIndex = computeShader.FindKernel(biasAndActivationkernelNames[(int)activationFunction]);
            }
            backPropZeroKernalIndex= computeShader.FindKernel(backPrepegateZeroStartKernalName);
            backPropAlterKernalIndex= computeShader.FindKernel(backPrepegateKernalNames[(int)activationFunction]);
            BackPropegateBiasPassKernalIndex = computeShader.FindKernel(backPrepegateMultiPassKernalNames[(int)activationFunction]);
            BackPropegateInputPassKernalIndex = computeShader.FindKernel(BackPropegateInputPassName);


            //BackPropegateBiasPassKernalIndex = computeShader.FindKernel("BackPropegateBiasPass");
            //BackPropegateInputSinglePassKernalIndex = computeShader.FindKernel("BackPropegateBiasPass");

            // Create buffers for inputs, outputs, weights, and biases during initialization
            Debug.Log("Initializing Layer Buffers " + ((inputBuffer == null) ? "creating new inputBuffer":"provided existing buffer for input"));
            if (inputBuffer == null) inputBuffer = new ComputeBuffer(NumInputs, sizeof(float)); //if input buffer was not passed in, create one
            outputBuffer = new ComputeBuffer(NumNeurons, sizeof(float));
            weightsBuffer = new ComputeBuffer(NumNeurons * NumInputs, sizeof(float));
            biasesBuffer = new ComputeBuffer(NumNeurons, sizeof(float));

            //if(propegatedErrorBuffer == null) 
                propegatedErrorBuffer = new ComputeBuffer(NumInputs, sizeof(float));
            activationDerivativeBuffer = new ComputeBuffer(NumNeurons, sizeof(float));
            sourceErrorBuffer = new ComputeBuffer(NumNeurons, sizeof(float));

            //assign compute buffers to compute shader
            if (!computeShaderIsSinglePass)
            {
                computeShader.SetBuffer(computeWeightSumKernalIndex, "inputBuffer", inputBuffer);

                computeShader.SetBuffer(computeWeightSumKernalIndex, "weightBuffer", weightsBuffer);
                computeShader.SetBuffer(computeAddBiasAndActivationKernelIndex, "biasBuffer", biasesBuffer);

                computeShader.SetBuffer(computeAddBiasAndActivationKernelIndex, "outputBuffer", outputBuffer);
                computeShader.SetBuffer(computeWeightSumKernalIndex, "outputBuffer", outputBuffer);
                computeShader.SetBuffer(resetZeroOutputKernalIndex, "outputBuffer", outputBuffer);
            }
            else
            {
                computeShader.SetBuffer(singlePassComputeLayerKernalIndex, "weightBuffer", weightsBuffer);
                computeShader.SetBuffer(singlePassComputeLayerKernalIndex, "inputBuffer", inputBuffer);
                computeShader.SetBuffer(singlePassComputeLayerKernalIndex, "biasBuffer", biasesBuffer);
                computeShader.SetBuffer(singlePassComputeLayerKernalIndex, "outputBuffer", outputBuffer);
            }

            /////  Set compute shader values and inputs.
            computeShader.SetInt("numInputs", NumInputs);
            computeShader.SetInt("numNeurons", NumNeurons);

            SetBackPropBuffers(backPropZeroKernalIndex);
            SetBackPropBuffers(BackPropegateBiasPassKernalIndex);
            SetBackPropBuffers(BackPropegateInputPassKernalIndex);
            
            SetShaderWeightsAndBiasData();
        }
        void SetBackPropBuffers(int kernelID)
        {
            computeShader.SetBuffer(kernelID, "progegatedErrorBuffer", propegatedErrorBuffer);
            if(kernelID!=backPropZeroKernalIndex)
                computeShader.SetBuffer(kernelID, "activationDerivativeBuffer", activationDerivativeBuffer);


            computeShader.SetBuffer(kernelID, "progegatedErrorBuffer", propegatedErrorBuffer);
            computeShader.SetBuffer(kernelID, "sourceErrorBuffer", sourceErrorBuffer);
            computeShader.SetBuffer(kernelID, "weightBuffer", weightsBuffer);
            computeShader.SetBuffer(kernelID, "biasBuffer", biasesBuffer);
            computeShader.SetBuffer(kernelID, "inputBuffer", inputBuffer);
            computeShader.SetBuffer(kernelID, "outputBuffer", outputBuffer);
        }
        private bool disposed = false;

        public void Dispose()
        {
            Dispose(true);
            System.GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    inputBuffer?.Dispose();
                    outputBuffer?.Dispose();
                    weightsBuffer?.Dispose();
                    biasesBuffer?.Dispose();

                    propegatedErrorBuffer?.Dispose();
                    sourceErrorBuffer?.Dispose();
                }
                disposed = true;
            }
        }
        void SetShaderWeightsAndBiasData()
        {
            //// assign values/initialize buffer data
            // Set the weights in the weights buffer (flatten the weights array)
            weightsBuffer.SetData(weights);
            // Set the biases in the biases buffer
            biasesBuffer.SetData(biases);
            // we don't need to set data for the output buffer, we will GET it after processing.
            // we don't need to set input data here, that will be done just before processing.

        }

        // Override the ComputeOutputs method to use the compute shader for processing
        // this variant set and gets the data on the gpu, for cpu use
        // it is intended to be used on all layers AFTER the first layer of the NeuralNet
        async public override UniTask<float[]> ComputeOutputs(float[] inputs)
        {
            this.inputs = inputs;
            if (inputs.Length != NumInputs)
                throw new System.ArgumentException("Invalid number of inputs passed to ComputeShaderLayer.ComputeOutputs.");

            // Set the inputs in the input buffer
            inputBuffer.SetData(inputs);

            DispatchComputeShader();
            await GetLastComputedOutput();

            return outputs;
        }
        // Override the ComputeOutputs method to use the compute shader for processing
        // this variant does NOT set and DOES gets the data on the gpu, for cpu use
        // it is intended to be used on all layers AFTER the first layer of the NeuralNet
        async public UniTask<float[]> ComputeOutputs()
        {
            DispatchComputeShader();
            await GetLastComputedOutput();
            return outputs;
        }

        //this variant assigns to, and then leaves the data on the gpu, and does not get it for cpu use after computation
        // it is intended to be used on the first layer of the NeuralNet
        async public UniTask ComputeLayer(float[] inputs)
        {
            if (inputs.Length != NumInputs)
                throw new System.ArgumentException("Invalid number of inputs passed to ComputeShaderLayer.ComputeOutputs.");
            this.inputs = inputs;
            // Set the inputs in the input buffer
            inputBuffer.SetData(inputs);
            DispatchComputeShader();
            return;
        }
        //this variant does not even assign data to the gpu, and it leves the data there, not getting it for cpu use after computation
        // it is intended to be used on all layers AFTER the first layer of the NeuralNet
        async public UniTask ComputeLayer()
        {
            DispatchComputeShader();
            return;
        }
        // specifically get data from the GPU for gpu use, inteded for use with ComputeLayer functions. ComputeOutputs functions do this automatically, every time.
        // it is intended to be called on the last layer of a neural net that uses the ComputeLayer functions, to get the final output as a float[]
        async public UniTask<float[]> GetLastComputedOutput()
        {
            await GetFloatArrayFromBuffer(outputBuffer, outputs);
            //lastOutputs = outputs;
            return outputs;
        }
        async public UniTask GetGPUData()
        {
            if (requests.Count > 0) return; //already in the process of waiting for requested data
            RequestGPUData();
            await WaitForGPUData();
        }
        List<UnityEngine.Rendering.AsyncGPUReadbackRequest> requests = new List<UnityEngine.Rendering.AsyncGPUReadbackRequest>();  //list of requests, one for each buffer
        public void RequestGPUData()
        {
            
            // Add all the readback requests to the list
            requests.Add(UnityEngine.Rendering.AsyncGPUReadback.Request(inputBuffer));
            requests.Add(UnityEngine.Rendering.AsyncGPUReadback.Request(outputBuffer));
            requests.Add(UnityEngine.Rendering.AsyncGPUReadback.Request(weightsBuffer));
            requests.Add(UnityEngine.Rendering.AsyncGPUReadback.Request(biasesBuffer));
            requests.Add(UnityEngine.Rendering.AsyncGPUReadback.Request(propegatedErrorBuffer));
            requests.Add(UnityEngine.Rendering.AsyncGPUReadback.Request(sourceErrorBuffer));
        }
        async public UniTask WaitForGPUData()
        {
            // Wait for all requests to complete
            int requestsCount = requests.Count;
            for(int i=0;i<requestsCount;i++)
            {
                while (!requests[i].done)
                    await UniTask.Yield();
            }
            
            if (inputs == null)
                inputs = new float[NumInputs];
            inputBuffer.GetData(inputs);
            outputBuffer.GetData(outputs);
            weightsBuffer.GetData(weights);
            biasesBuffer.GetData(biases);
            propegatedErrorBuffer.GetData(propagatedErrors);
            sourceErrorBuffer.GetData(sourceErrors);
            requests.Clear();
        }

        //not efficient at getting more than one buffer at a time since it yields until the request is ready.
        async UniTask GetFloatArrayFromBuffer(ComputeBuffer buffer,float[] array)
        {
          //  Debug.Log("getting buffer data");
            //bool ready = false;
            UnityEngine.Rendering.AsyncGPUReadbackRequest request = UnityEngine.Rendering.AsyncGPUReadback.Request(buffer);//, (x) => { ready = true; });
            
            while (!request.done)
              await UniTask.Yield();
            buffer.GetData(array);
          //  Debug.Log("got buffer data");
        }
        
        /// <summary>
        /// This function dispatches the compute shaders
        /// </summary>
        /// <returns></returns>
        void DispatchComputeShader()
        {
            if (inputBuffer == null) throw new System.ArgumentException("Shader layer invoked without input buffer assigned.  Call ComputeLayer(float[] inputs) to pass in input values, or create the layer with an input buffer provided.");
            if (computeShader == null)
            {
                Debug.Log("Null ComputeShader Aborting Dispatch");
                return;
            }
            // Set the compute shader kernel and dispatch it
            if (!computeShaderIsSinglePass)
            {
                /*UnityEngine.Rendering.CommandBuffer commandBuffer = new UnityEngine.Rendering.CommandBuffer();
                commandBuffer.DispatchCompute(computeShader, resetZeroOutputKernalIndex, NumNeurons, 1, 1);//first we zero the output buffer contents
                commandBuffer.DispatchCompute(computeShader, computeWeightSumKernalIndex, NumNeurons * NumInputs, 1, 1);//next we compute all the input values * connecting weights, summing the results to its neuron ouput
                commandBuffer.DispatchCompute(computeShader, computeAddBiasAndActivationKernelIndex, NumNeurons, 1, 1);//next we apply the activation function and neuron's bias to the inputsum and so, generate the final output of eah neuron activation upon input sum
                Graphics.ExecuteCommandBuffer(commandBuffer);
                commandBuffer.Release();*/
                //the multi pass method allows for greater parallelization as all input connects can be processed in parallel
                computeShader.Dispatch(resetZeroOutputKernalIndex, NumNeurons, 1,1);//first we zero the output buffer contents
                computeShader.Dispatch(computeWeightSumKernalIndex, NumNeurons * NumInputs, 1,1);//next we compute all the input values * connecting weights, summing the results to its neuron ouput
                computeShader.Dispatch(computeAddBiasAndActivationKernelIndex, NumNeurons, 1, 1);//next we apply the activation function and neuron's bias to the inputsum and so, generate the final output of eah neuron activation upon input sum
            }
            else // the single pass method can paralleleize neurons, but not the connections to those neurons (must loop for each neuron)
            {
                //Debug.Log("Dispatching kernel:" + singlePassComputeLayerKernalIndex);
                computeShader.Dispatch(singlePassComputeLayerKernalIndex, NumNeurons, 1, 1);
            }
           // await Task.Yield();
            return;
        }
        async public override UniTask<float[]> Backpropagate(float[] inputs, float[] errors, float learningRate)
        {
            if (computeShader == null) return new float[0];
            computeShader.SetFloat("learningRate", learningRate);
            sourceErrors = errors;
            sourceErrorBuffer.SetData(sourceErrors);
            computeShader.Dispatch(backPropZeroKernalIndex, NumInputs, 1, 1);
            
            computeShader.Dispatch(BackPropegateBiasPassKernalIndex, NumNeurons, 1, 1);
            //computeShader.Dispatch(BackPropegateInputPassKernalIndex, NumNeurons*NumInputs, 1, 1);
            computeShader.Dispatch(BackPropegateInputPassKernalIndex, NumInputs,1, 1);

            //computeShader.Dispatch(backPropAlterKernalIndex, NumNeurons, 1, 1);
            await GetGPUData();
            /*
            //float[] ret = new float[NumInputs];
            await GetFloatArrayFromBuffer(inputErrorBuffer, propagatedErrors);
            //await GetFloatArrayFromBuffer(weightsBuffer, weights);
            weightsBuffer.GetData(weights);
            await GetFloatArrayFromBuffer(biasesBuffer, biases);
            await GetFloatArrayFromBuffer(outputErrorBuffer, sourceErrors);
            //biasesBuffer.GetData(biases);
            */

            //float[] ret=base.Backpropagate(inputs, errors, learningRate);
            //SetShaderWeightsAndBiasData();//we modified these with backpropegate- update data on GPU
            return propagatedErrors;
        }
    }
}