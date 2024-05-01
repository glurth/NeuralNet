using System.Collections.Generic;
using UnityEngine;
using Cysharp.Threading.Tasks;// System.Threading.Tasks;

namespace EyE.NNET
{
    /// <summary>
    /// This variant of a NeuralNet is intended to use ComputeShaderLayer's rather than NetLayers.  While the base class would work with them, this variant allows for greater control over when data is passed to/from the GPU.  
    /// It also implements IDisposable to handle cleaning up of buffers.
    /// </summary>
    public class NeuralNetComputeShader : NeuralNet,System.IDisposable
    {
        ComputeShader layerComputeShader;
        bool computeShaderIsSingleThreaded;
        public NeuralNetComputeShader(int numInput, int numOutput, ComputeShader layerComputeShader, bool computeShaderIsSingleThreaded) : base(numInput, numOutput)
        {
            this.computeShaderIsSingleThreaded = computeShaderIsSingleThreaded;
            this.layerComputeShader = layerComputeShader;
        }
        public NeuralNetComputeShader(NeuralNet source, ComputeShader layerComputeShader, bool computeShaderIsSingleThreaded) : base(source.NumInputs, source.NumOutputs)
        {
            ComputeShaderLayer lastLayer = null;
            foreach (NetLayer layer in source.Layers)
            {
                ComputeShaderLayer newLayer = new ComputeShaderLayer(layerComputeShader, layer, computeShaderIsSingleThreaded, lastLayer);
                lastLayer = newLayer;
                layers.Add( newLayer);
            }
            this.computeShaderIsSingleThreaded = computeShaderIsSingleThreaded;
            this.layerComputeShader = layerComputeShader;
        }
        ComputeBuffer outputComputeBuffer;
        public override void PopulateLayersRandomly(ActivationFunction alwaysUse, int numHiddenLayers = -1, int layerSizeMin = -1, int layerSizeMax = -1)
        {
            if (layerSizeMin == -1) layerSizeMin = Mathf.Min(NumInputs, NumOutputs) + 1;
            if (layerSizeMax == -1) layerSizeMax = Mathf.Max(NumInputs, NumOutputs) * 2;

            int numLayers = UnityEngine.Random.Range(2, 4);
            if (numHiddenLayers != -1)
                numLayers = numHiddenLayers + 1;
            int previousLayerNumOutputs = NumInputs;
            this.layers = new List<NetLayer>();
            ComputeShaderLayer lastLayer = null; 
            for (int i = 0; i < numLayers; i++)
            {
                int layerNumOutputs = Random.Range(layerSizeMin, layerSizeMax);
                if (i == numLayers - 1) layerNumOutputs = NumOutputs;
                ComputeShaderLayer newLayer;
                if (i < numLayers - 1)// && i>0)
                {
                    
                    newLayer = new ComputeShaderLayer(layerComputeShader, layerNumOutputs, previousLayerNumOutputs, alwaysUse, computeShaderIsSingleThreaded, lastLayer);
                    // lastLayerOutputBuffer = newLayer.outputBuffer;
                }
                else //output layer
                {
                    
                    newLayer = new ComputeShaderLayer(layerComputeShader, layerNumOutputs, previousLayerNumOutputs, ActivationFunction.None, computeShaderIsSingleThreaded, lastLayer);
                    //  lastLayerOutputBuffer = newLayer.outputBuffer;
                }

                this.layers.Add(newLayer);
                previousLayerNumOutputs = layerNumOutputs;
                lastLayer = newLayer;
            }
            outputComputeBuffer = lastLayer.outputBuffer;
        }

        public void Dispose()
        {
            foreach (NetLayer netLayer in layers)
                (netLayer as ComputeShaderLayer).Dispose();
        }
        async public UniTask GetGPUData()
        {
            foreach (NetLayer netLayer in layers)
            {
                ComputeShaderLayer layer = netLayer as ComputeShaderLayer;
                layer.RequestGPUData();
            }
            foreach (NetLayer netLayer in layers)
            {
                ComputeShaderLayer layer = netLayer as ComputeShaderLayer;
                await layer.WaitForGPUData();
            }

        }
        async public virtual UniTask<float[]> GPUThink(float[] input)
        {

            if (input.Length != NumInputs)
            {
                Debug.Log("NeuralNet Think operation failed.  Input size does not match network configuration.  Given: " + input.Length + " Expected: " + NumInputs);
                throw new System.ArgumentException("Input size does not match network configuration.  Given: " + input.Length + " Expected: " + NumInputs);
            }
            _lastInputs = input;
            //Debug.Log("NeuralNet starting Think operation. Inputs:"+numInput+ "  HiddenLayers:" + (layers.Count-1) + "outputs: " + numOutput);
            int count = 0;
            foreach (NetLayer netLayer in layers)
            {
                ComputeShaderLayer layer = netLayer as ComputeShaderLayer;
                if (count == 0)//first layer give inputs
                    layer.ComputeLayer(input);
                else
                    layer.ComputeLayer();
                if (count == layers.Count - 1)//last layer get outputs
                    lastOutputs = await layer.GetLastComputedOutput();
                count++;
            }
            //Debug.Log("    NeuralNet processing complete");
            return lastOutputs;// UniTask.FromResult<float[]>(activations);
        }
    }
}
