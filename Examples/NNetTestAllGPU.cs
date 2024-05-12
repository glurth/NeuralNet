using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using EyE.NNET;

public class NNetTestAllGPU : MonoBehaviour
{
    public float input0 = 1;
    public float input1 = 1;
    public float inputOp = 0;

    private NeuralNetComputeShader nnet;
    public NeuralNetComputeShader Nnet { get => nnet; set => nnet = value; }

    public Visualizer display1;

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

        nnet = new NeuralNetComputeShader(2,1, layerComputeShader, singlePassComputeShader);
        nnet.PopulateLayersRandomly(funcToUse, 2, 4, 5);
        if (display1 != null)
        {
            display1.neuralNetwork = nnet;
           // display1.useLog = useLog;
        }

        if (useLog)
        {
            Debug.Log(nnet.ToString());
            Debug.Log("*****Start DONE **********");
        }
    }
    private void OnDestroy()
    {
        if(useLog)
            Debug.Log("Disposing");
        nnet?.Dispose();
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
        float[] output = await nnet.GPUThink(inputs);
        //float[] output = await nnet.Think(inputs);//.Result;
        if (useLog) Debug.Log("Think cycle " + cycleCounter + " Dispatched");

        //goal output is (input0 + input1)
        float error;// = output[0] - (input0 + input1);
      //  if (inputOp < 1)
            error = output[0] - (input0 + input1);
      //  else
      //      error = output[0] - (input0 - input1);
        errors[0] = error;
        await nnet.Backpropagate(errors, learningRate);

        if (useLog) Debug.Log("Backpropagate 1 cycle " + cycleCounter + " done. errors: " + string.Join(",", errors));// + nnet.ToString());

        thinkInProgress = false;
        // Debug.Log(name + " Think cycle " + cycleCounter + "complete-  Input0:" + input0 + "  input1:" + input1 + "   output: " + output[0] + "  error:" + error);
    }
}
