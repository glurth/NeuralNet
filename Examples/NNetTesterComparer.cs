using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using EyE.NNET;

public class NNetTesterComparer : MonoBehaviour
{
    public float input0 = 1;
    public float input1 = 1;
    public float inputOp = 0;

    public int numNetLayers = 2;
    public int minNeuronsPerLayer = 4;
    public int maxNeuronsPerLayer = 5;

    private NeuralNet nnet;
    public NeuralNet Nnet { get => nnet; set => nnet = value; }

    private NeuralNetComputeShader nnet2;
    public NeuralNetComputeShader Nnet2 { get => nnet2; set => nnet2 = value; }

    public Visualizer display1;
    public Visualizer display2;
    public bool continuousThink = true;
    public bool singleThink = false;
    public ActivationFunction funcToUse;
    public float learningRate = 0.01f;
    int cycleCounter = 0;
    public ComputeShader layerComputeShader;
    public bool singlePassComputeShader=true;
    public bool useLog = true;
    // Start is called before the first frame update
    void Start()
    {
        nnet = new NeuralNet(2, 1);
        nnet.PopulateLayersRandomly(funcToUse, numNetLayers, minNeuronsPerLayer, maxNeuronsPerLayer);
        nnet2 = new NeuralNetComputeShader(nnet, layerComputeShader, singlePassComputeShader);
        if (display1 != null)
        {
            display1.neuralNetwork = nnet;
           // display1.useLog = useLog;
        }
        if (display2 != null)
        {
            display2.neuralNetwork = nnet2;
           // display2.useLog = useLog;
        }
        if (useLog)
        {
            Debug.Log(nnet.ToString());
            Debug.Log(nnet2.ToString());
            Debug.Log("*****Start DONE **********");
        }
    }
    private void OnDestroy()
    {
        if(useLog)
            Debug.Log("Disposing");
        nnet2?.Dispose();
    }
    public int processUntilCycle =-1;
    // Update is called once per frame
    void Update()
    {
        
        for (int i = 0; i < 30; i++)
        {
            if (processUntilCycle == -1 || processUntilCycle >= cycleCounter)
            {
                if (continuousThink || singleThink)
                    DoThink();
            }
            else
                continuousThink = false;
            singleThink = false;
        }
    }
    public int changeAfterNumAttempts = 100;
    int changeCounter = 0;
    float[] inputs= new float[2];
    float[] errors = new float[1];
    float[] errors2 = new float[1];

    bool thinkInProgress = false;
    
    async void DoThink()
    {
        if (thinkInProgress) return;
        thinkInProgress = true;
        cycleCounter++;

        if (changeCounter++ > changeAfterNumAttempts)
        {
            input0 = Random.Range(-50, 50);
            input1 = Random.Range(-50, 50);
            inputOp = Random.Range(0f, 2f);
            changeCounter = 0;
        }
        inputs[0] = input0;
        inputs[1] = input1;
      //  inputs[2] = inputOp;
        float[] output = await nnet.Think(inputs);//.Result;

        if (useLog) Debug.Log("Think 1 cycle "+cycleCounter +" done: " + nnet.ToString());
        //goal output is (input0 + input1)
        /*float error;// = output[0] - (input0 + input1);
      //  if (inputOp < 1)
            error = output[0] - (input0 + input1);
      //  else
      //      error = output[0] - (input0 - input1);
        errors[0] = error;*/
        errors = ComputeErrors(inputs, output);
        nnet.Backpropagate(errors, learningRate);
        if (useLog) Debug.Log("Backpropagate 1 cycle " + cycleCounter + " done. errors: " + string.Join(",",errors) + nnet.ToString());

        float[] output2 = await nnet2.GPUThink(inputs);//

        if (useLog)
        {
            await nnet2.GetGPUData();
            Debug.Log("Think 2 cycle " + cycleCounter + " done: " + nnet2.ToString());
        }
        //goal output is (input0 + input1)
     /* //   if(inputOp<1)
            error = output2[0] - (input0 + input1);
    //    else
    //        error = output2[0] - (input0 - input1);
        errors2[0] = error;*/
        errors2 = ComputeErrors(inputs, output2);
        await nnet2.Backpropagate(errors, learningRate);//  await is test
        if (useLog)
        {
            await nnet2.GetGPUData();
            Debug.Log("Backpropagate 2 cycle " + cycleCounter + " done. errors: " + string.Join(",", errors2) + nnet2.ToString());
        }

        thinkInProgress = false;
        // Debug.Log(name + " Think cycle " + cycleCounter + "complete-  Input0:" + input0 + "  input1:" + input1 + "   output: " + output[0] + "  error:" + error);
    }

    float[] ComputeErrors(float[] input, float[] output)
    {
    //    if (input[2] < 1) //add
            return new float[] { output[0] - (input[0] + input[1]) };
     //   else //subtract 
     //       return new float[] { output[0] - (input[0] - input[1]) };
    }
}
