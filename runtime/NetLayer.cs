using Cysharp.Threading.Tasks;// System.Threading.Tasks;
using System.IO;
using UnityEngine;
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
            biasErrors = new float[numNeurons];
            weightsErrors = new float[numNeurons, numInputs];
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
            biasErrors = new float[numNeurons];
            weightsErrors = new float[numNeurons, numInputs];
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

        public float[] biasErrors; //Error in the biases of each neuron on the layer
        public float[,] weightsErrors;// Error in the weight of each connection from each neuron.  [neuronInexInLayer,connectionIndexInNeuron]
        public float[] propagatedErrors;// Errors to be propagated to the previous layer, one for each neuron on the previous layer.
        public float[] sourceErrors;// Record of the last errors in passed into backpropegate function- one for each output/neuron in the layer.

        /// <summary>
        /// Computes the changes to weight and biases in this layer, and returns the errors that should be passed to the previous layer (previous layer because: the neuron net will call this function for the layers in reverse order).
        /// This function used cashed values from the previous think operation, such as lastInputs and lastOutputs.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="errors"></param>
        /// <param name="learningRate"></param>
        /// <param name="gradientThreshold">propegated errors will be clipped to this max absolute value</param>
        /// <returns></returns>
        public virtual UniTask<float[]> Backpropagate(float[] inputs, float[] errors, float learningRate,float gradientThreshold=0)
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
                biasErrors[i] = errors[i] * lastOutputActivationDerivative;
                // Update biases
                biases[i] -= learningRate * biasErrors[i];

                // Compute the gradient for weights
                for (int j = 0; j < numInputs; j++)
                {
                    weightsErrors[i, j] = errors[i] * lastOutputActivationDerivative * inputs[j];

                    // Accumulate errors to be propagated to the previous layer
                    propagatedErrors[j] += errors[i] * lastOutputActivationDerivative * weights[i, j];

                    // Apply gradient clipping if requested
                    if (gradientThreshold > 0)
                    {
                        if (weightsErrors[i, j] > gradientThreshold)
                        {
                            weightsErrors[i, j] = gradientThreshold;
                        }
                        else if (weightsErrors[i, j] < -gradientThreshold)
                        {
                            weightsErrors[i, j] = -gradientThreshold;
                        }
                    }

                    //update weights
                    weights[i, j] -= learningRate * weightsErrors[i, j];
                }
            }
            return new UniTask<float[]> (propagatedErrors);
        }


        /////////////////////
        ////// Serialization
        /////////////////////
        
        public NetLayer(int numInputs,int numNeurons, float[,] weights, float[] biases, ActivationFunction activationFunction)
        {
            this.numInputs = numInputs;
            this.numNeurons = numNeurons;
            this.weights = weights;
            this.biases = biases;
            this.activationFunction = activationFunction;
        }

        public void SerializeBinary(BinaryWriter writer)
        {
            writer.Write(numInputs);
            writer.Write(numNeurons);
            // Serialize weights
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    writer.Write(weights[i, j]);
                }
            }
            // Serialize biases
            foreach (var bias in biases)
            {
                writer.Write(bias);
            }
            // Serialize activationFunction
            writer.Write((int)activationFunction);
        }

        public static NetLayer DeserializeBinary(BinaryReader reader)
        {
            int numInputs = reader.ReadInt32();
            int numNeurons = reader.ReadInt32();

            // Deserialize weights
            float[,] weights = new float[numNeurons, numInputs];
            for (int j = 0; j < numNeurons; j++)
            {
                for (int k = 0; k < numInputs; k++)
                {
                    weights[j, k] = reader.ReadSingle();
                }
            }
            // Deserialize biases
            float[] biases = new float[numNeurons];
            for (int j = 0; j < numNeurons; j++)
            {
                biases[j] = reader.ReadSingle();
            }
            // Deserialize activationFunction
            ActivationFunction activationFunction = (ActivationFunction)reader.ReadInt32();
            return new NetLayer(numInputs, numNeurons, weights, biases, activationFunction);
        }

    }
}