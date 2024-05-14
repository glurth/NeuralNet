using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cysharp.Threading.Tasks;// System.Threading.Tasks;
using System.IO;

namespace EyE.NNET
{
    //start test

    //integers and List<integers> contain indexes into allConnections and allNeurons

    //specifies the connection between a source and destination neuron
    public class Connection
    {
        public int sourceNeuron;
        public int destinationNeuron;
        public float weight;
        public ActivationFunction activationFunction;

        public float OutputValue(ConnectionNet cnet)
        {
            return activationFunction.Activate(cnet.allNeurons[sourceNeuron].outputValue) * weight;
        }
    }
    public class Neuron
    {
        public float inputValue;
        public float outputValue;

        public float bias;
        [System.NonSerialized]
        public List<int> inputConnections;
        [System.NonSerialized]
        public List<int> outputConnections;
        public int sequenceIndex; //distance from start, aka "layer"
        public int elementInSequence;// distance into current sequence/layer of this neuron
        public void ComputeValueFromInputs(ConnectionNet cnet)
        {
            float valSum = 0;
            foreach (int index in inputConnections)
            {
                valSum += cnet.allConnections[index].OutputValue(cnet);
            }
            outputValue = valSum + bias;
        }
    }

    public class ConnectionNet
    {
        //a list of all the connections between neurons in the cnet
        public List<Connection> allConnections;
        //a list of all neurons in the Cnet
        public List<Neuron> allNeurons;

        public List<List<int>> nueronsByLayer;
        /// <summary>
        /// list of indexes into `allNeurons`, specifies which neurons have no inputs- these will be fed by the user when asking the CNet to compute
        /// </summary>
        public List<int> inputNeurons;
        
        /// <summary>
        /// list of indexes into `allNeurons`, specifies which neurons have no inputs- these will be read by the user after asking the CNet to compute
        /// </summary>
        public List<int> outputNeurons;

        List<List<int>> ComputeNeuronSequenceIndexs()
        {
            List<List<int>> neuronIndeBylayers = new List<List<int>>();
            int seqCounter = 0;
            List<int> currentLayer = new List<int>(inputNeurons);
            List<int> nextLayer = new List<int>();
            while (currentLayer.Count > 0)
            {
                for (int i = 0; i < currentLayer.Count; i++)
                {
                    int nIndex = currentLayer[i];
                    allNeurons[nIndex].sequenceIndex = seqCounter; //will be updated with 
                    foreach (int outConnIndex in allNeurons[nIndex].outputConnections)
                    {
                        nextLayer.Add(allConnections[outConnIndex].destinationNeuron);
                    }
                }
                currentLayer = nextLayer;
                seqCounter++;
                neuronIndeBylayers.Add(new List<int>());
            }
            for(int nIndex =0; nIndex<allNeurons.Count; nIndex++)
            {
                Neuron n = allNeurons[nIndex];
                List<int> layerList = neuronIndeBylayers[n.sequenceIndex];
                n.elementInSequence = layerList.Count;
                layerList.Add(nIndex);
            }
            return neuronIndeBylayers;
        }

        public void OnDeserialize()
        {
            RebuildConnections();
            inputNeurons = NeuronsWithNoInputConnections();
            outputNeurons = NeuronsWithNoOutputConnections();

            /// <summary>
            /// Rebuilds inputConnections and outputConnections for all neurons based on the connections present in the allConnections list.
            /// </summary>
            void RebuildConnections()
            {
                for (int i = 0; i < allNeurons.Count; i++)
                {
                    Neuron neuron = allNeurons[i];
                    neuron.inputConnections = new List<int>();
                    neuron.outputConnections = new List<int>();

                    for (int j = 0; j < allConnections.Count; j++)
                    {
                        Connection connection = allConnections[j];

                        if (connection.sourceNeuron == i)
                        {
                            neuron.outputConnections.Add(connection.destinationNeuron);
                        }

                        if (connection.destinationNeuron == i)
                        {
                            neuron.inputConnections.Add(connection.sourceNeuron);
                        }
                    }
                }
            }
            //generate list of input neurons (assumes RebuildConnections has been run)
            List<int> NeuronsWithNoInputConnections()
            {
                List<int> neuronsWithNoInput = new List<int>();

                // Iterate through all neurons to find those with no input connections
                for (int i = 0; i < allNeurons.Count; i++)
                {
                    if (allNeurons[i].inputConnections.Count == 0)
                    {
                        neuronsWithNoInput.Add(i);
                    }
                }

                return neuronsWithNoInput;
            }
            //generate list of output neurons (assumes RebuildConnections has been run)
            List<int> NeuronsWithNoOutputConnections()
            {
                List<int> neuronsWithNoOutput = new List<int>();

                // Iterate through all neurons to find those with no input connections
                for (int i = 0; i < allNeurons.Count; i++)
                {
                    if (allNeurons[i].outputConnections.Count == 0)
                    {
                        neuronsWithNoOutput.Add(i);
                    }
                }

                return neuronsWithNoOutput;
            }
        }
        void Think(float[] inputs)
        {
            List<Neuron> currentSeq = new List<Neuron>();
            List<Connection> currentSeqConnections = new List<Connection>();
            int count = 0;
            foreach (int index in inputNeurons)
            {
                Neuron n = allNeurons[index];
                n.outputValue = inputs[count++];
                //currentSeq.Add(n.outputConnections);
            }
            foreach (Neuron n in allNeurons)
            {
                n.ComputeValueFromInputs(this);
            }



        }
    }
    public class Sequence
    {
        public List<Neuron> neurons;
        public List<int> connectionInputs;

        static List<Sequence> GenerateSequences(ConnectionNet cnet)
        {
            List<Sequence> seqs = new List<Sequence>();

            return seqs;
        }
        void ComputeSequence(ConnectionNet cnet)
        {
            for (int c = 0; c < connectionInputs.Count; c++)
            {
                Connection conn = cnet.allConnections[c];
                cnet.allNeurons[conn.destinationNeuron].inputValue = cnet.allNeurons[conn.destinationNeuron].bias;
            }
            for (int c = 0; c < connectionInputs.Count; c++)
            {
                Connection conn = cnet.allConnections[c];
                float value = cnet.allNeurons[conn.sourceNeuron].outputValue;
                value = conn.activationFunction.Activate(value);
                value *= conn.weight;
                cnet.allNeurons[conn.destinationNeuron].inputValue += value;
            }
        }
    }
    //end test

    //this basic NeuralNet uses static layers of neurons.  Every neuron has a connection to every neuron in the previous layer.  The asynchronous `Think` function may be called to process a set of inputs.
    //the net stores a list of NetLayer which contain details about the neurons in a given layer.  The order of this list defines the sequence in which layers are processed, one after another.
    // the net also stores the most recently processed (`Think`) input and output values, for use by post-think, optional, backpropegation(`Backpropagate`).
    /// <summary>
    /// This basic NeuralNet uses static layers of neurons.  Every neuron has a connection to every neuron in the previous layer.  
    /// The asynchronous `Think` function may be called to process a set of inputs.
    /// The net stores a list of NetLayer which contain details about the neurons in a given layer, and their connections to neurons of the previous layers.  
    ///     The order of this list defines the sequence in which layers are processed, one after another.  
    ///     It is important to note this list must remain static, (particularly it's sequence, and number of elements in each layer) because neurons are defined with a specific number input connections, one for each neuron in the previous layer.
    /// The net also stores the most recently processed (`Think`) input and output values, for use by post-think, optional, backpropegation(`Backpropagate`).
    /// ToString() will output all the information about the net, including various details of each neuron of each layer.  Be warned, this can grow large and slow to process.  It also required data must be on the CPU, and transfered from the GPU, if there like with ComputeShaderLayers)
    /// </summary>
    [System.Serializable]
    public class NeuralNet
    {

        [SerializeField]
        protected int numInput;
        public int NumInputs => numInput;
        [SerializeField]
        protected int numOutput;
        public int NumOutputs => numOutput;
        [SerializeField]
        protected List<NetLayer> layers;
        public IReadOnlyList<NetLayer> Layers
        {
            get { return layers; }
        }

        public NeuralNet Clone()
        {
            NeuralNet clonedNet = new NeuralNet(numInput, numOutput);// (NeuralNet)this.MemberwiseClone();
            foreach (NetLayer layer in this.layers)
            {
                clonedNet.layers.Add(layer.Clone());

            }

            return clonedNet;
        }

        public NeuralNet CloneAndMutateLayers(float addLayerMutationPerLayerChance, float activationFunctionChangeChance, float biasMutationChance, float biasMutationAmount, float numNeuronsMutationChance, float numNeuronsMutationAmount, float weightsMutationChance, float weightsMutationAmount)
        {

            NeuralNet clonedNet = new NeuralNet(numInput, numOutput);// (NeuralNet)this.MemberwiseClone();
            foreach (NetLayer layer in this.layers)
            {
                clonedNet.layers.Add(layer.CloneAndMutate(activationFunctionChangeChance, biasMutationChance, biasMutationAmount, numNeuronsMutationChance, numNeuronsMutationAmount, weightsMutationChance, weightsMutationAmount));
                if (Random.value < addLayerMutationPerLayerChance)
                {
                    //if we add a random new layer after this one.. it will need numInputs == thislayer's numOutputs..  and it's numOutputs == thislayer's numOutputs
                    clonedNet.layers.Add(new NetLayer(layer.NumNeurons, layer.NumNeurons, ActivationFunctionExtension.RandomActivationFunction()));
                }
            }
            return clonedNet;
        }

        public NeuralNet(int numInput, int numOutput)
        {
            this.numInput = numInput;
            this.numOutput = numOutput;
            this.layers = new List<NetLayer>();
        }
        public NeuralNet(NeuralNet source)
        {
            this.numInput = source.numInput;
            this.numOutput = source.numOutput;
            this.layers = new List<NetLayer>();
            foreach (NetLayer layer in source.layers)
            {
                layers.Add(layer.Clone());
            }

        }

        /// <summary>
        /// Generates a Random NeuralNet.
        /// </summary>
        /// <param name="alwaysUse"></param>
        /// <param name="numHiddenLayers"></param>
        /// <param name="layerSizeMin"></param>
        /// <param name="layerSizeMax"></param>
        public virtual void PopulateLayersRandomly(ActivationFunction alwaysUse,int numHiddenLayers = -1, int layerSizeMin=-1, int layerSizeMax = -1)
        {
            if(layerSizeMin == -1) layerSizeMin= Mathf.Min(numInput, numOutput)+1;
            if( layerSizeMax == -1 ) layerSizeMax= Mathf.Max(numInput, numOutput) * 2;
            
            int numLayers = UnityEngine.Random.Range(2, 4);
            if (numHiddenLayers != -1)
                numLayers = numHiddenLayers + 1;
            int previousLayerNumOutputs = numInput;
            this.layers = new List<NetLayer>();
            for (int i = 0; i < numLayers; i++)
            {
                int layerNumOutputs = Random.Range(layerSizeMin, layerSizeMax);
                if (i == numLayers - 1) layerNumOutputs = numOutput;
                NetLayer newLayer;
                if (i < numLayers - 1)// && i>0)
                {
                    newLayer = new NetLayer(layerNumOutputs, previousLayerNumOutputs, alwaysUse);
                }
                else //output layer
                    newLayer = new NetLayer(layerNumOutputs, previousLayerNumOutputs, ActivationFunction.None);

                this.layers.Add(newLayer);
                previousLayerNumOutputs = layerNumOutputs;
            }
        }
        [SerializeField]
        protected float[] _lastInputs;
        public float[] lastInputs => _lastInputs;
        protected float[] lastOutputs;
        async public virtual UniTask<float[]> Think(float[] input)
        {

            if (input.Length != numInput)
            {
                Debug.Log("NeuralNet Think operation failed.  Input size does not match network configuration.  Given: " + input.Length + " Expected: " + numInput);
                throw new System.ArgumentException("Input size does not match network configuration.  Given: " + input.Length + " Expected: " + numInput);
            }
            _lastInputs = input;
            //Debug.Log("NeuralNet starting Think operation. Inputs:"+numInput+ "  HiddenLayers:" + (layers.Count-1) + "outputs: " + numOutput);
            float[] activations = input;
            foreach (NetLayer layer in layers)
            {
                activations = await layer.ComputeOutputs(activations);//.Result;
            }
            //Debug.Log("    NeuralNet processing complete");
            lastOutputs = activations;
            return activations;// UniTask.FromResult<float[]>(activations);
        }
        public float[] lastOutputErrors;
        async public UniTask Backpropagate(float[] outputErrors, float learningRate)
        {
            if (lastInputs == null)
            {
                Debug.Log("NeuralNet Backpropagation failed. At least one Think Process must be performed before backprpegation can be done.");
                return;
            }
            if (lastInputs.Length != numInput)
            {
                Debug.Log("NeuralNet Backpropagation failed. Input size does not match network configuration.");
                return;
            }
            lastOutputErrors = outputErrors;
            // Backpropagate errors through the layers
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                NetLayer layer = layers[i];

                // Calculate layer input
                float[] layerInput;
                if (i == 0)
                {
                    layerInput = lastInputs;
                }
                else
                {
                    NetLayer previousLayer = layers[i - 1];
                    layerInput = previousLayer.lastOutputs;
                }


                float[] nextLayerErrors = await layer.Backpropagate(layerInput, outputErrors, learningRate);
                //Debug.Log("Backprop for layer " + i + " completed.  nextLevelErrors: " + string.Join(",",nextLayerErrors));
                outputErrors = nextLayerErrors; // Update errors for the next iteration
            }
        }

        /// <summary>
        /// Will output all the information about the net, including various details of each neuron of each layer.  WARNING: this can grow large and slow to process.  It also required data must be on the CPU, or transfered from the GPU to the CPU (as with ComputeShaderLayers).
        /// </summary>
        /// <returns>formatted string proving the details of the net</returns>
        public override string ToString()
        {
            string s = "NeuralNet of Type: " + GetType() +
                "\nInputs[" + NumInputs + "]: " + (lastInputs == null ? "null":string.Join(",", lastInputs)) +
                "\nOutput[" + NumOutputs + "]: " + (lastOutputs == null ? "null" : string.Join(",", lastOutputs)) + 
                "\nLayers[" + layers.Count + "]:";
            int c = 0;
            foreach (NetLayer layer in layers)
            {
                s += "\n\tLayer " + c;
                s += "\n\tActivation Function: " + System.Enum.GetName(typeof(ActivationFunction),layer.activationFunction);
                s += "\n\tInputs: " + string.Join(",", layer.inputs == null ? "null" : string.Join(",", layer.inputs));
                s += "\n\tNeurons[" + layer.NumNeurons + "]: ";
                s += "\n\t\tBiases: " + string.Join(",", layer.biases);
                s += "\n\t\tBiasErrors:" + string.Join(",", layer.biasErrors);
                s += "\n\t\tOutputs: " + string.Join(",", layer.lastOutputs == null ? "null" : string.Join(",", layer.lastOutputs));
                s += "\n\t\tWeights:\n"+StringExtension.GenerateFloatTable(layer.weights,false,"\t\t");
                s += "\n\t\tWeightErrors (COL: input, ROW: neuron):\n" + StringExtension.GenerateFloatTable(layer.weightsErrors, false, "\t\t");
                s += "\n\t\tpropagatedErrors: " + string.Join(",", layer.propagatedErrors);
                s += "\n\t\tsourceErrors: " + string.Join(",", layer.sourceErrors);
                c++;
            }
            return s;
        }

        // Serialize the NeuralNet instance to a binary file
        public void SaveBinary(string filePath)
        {
            using (BinaryWriter writer = new BinaryWriter(File.Open(filePath, FileMode.Create)))
            {
                writer.Write(numInput);
                writer.Write(numOutput);
                writer.Write(layers.Count);
                
                foreach (var layer in layers)
                {
                    layer.SerializeBinary(writer);
                }
            }
        }

        // Deserialize a NeuralNet instance from a binary file
        public static NeuralNet LoadBinary(string filePath)
        {
            NeuralNet neuralNet = new NeuralNet(0, 0); // Initialize with default values
            using (BinaryReader reader = new BinaryReader(File.Open(filePath, FileMode.Open)))
            {
                neuralNet.numInput = reader.ReadInt32();
                neuralNet.numOutput = reader.ReadInt32();
                int layerCount = reader.ReadInt32();
                
                neuralNet.layers = new List<NetLayer>();
                for (int i = 0; i < layerCount; i++)
                {
                    neuralNet.layers.Add(NetLayer.DeserializeBinary(reader));
                }
            }
            return neuralNet;
        }
        // Serialize the NeuralNet instance to a JSON file
        public void SaveJson(string filePath)
        {
            string json = JsonUtility.ToJson(this);
            File.WriteAllText(filePath, json);
        }

        // Deserialize a NeuralNet instance from a JSON file
        public static NeuralNet LoadJson(string filePath)
        {
            string json = File.ReadAllText(filePath);
            return JsonUtility.FromJson<NeuralNet>(json);
        }
    }
}
