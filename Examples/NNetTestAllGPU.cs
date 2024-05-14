using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cysharp.Threading.Tasks;
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
    public int numNetLayers = 2;
    public int minNeuronsPerLayer = 4;
    public int maxNeuronsPerLayer = 5;
    public float learningRate = 0.01f;

    public int processUntilCycle = -1;
    public int changeAfterNumAttempts = 100;


    public ComputeShader layerComputeShader;
    public bool singlePassComputeShader=true;
    public bool useLog = true;

    public string saveFileName = "NNetSeri";
    public void DoSave()
    {
        nnet.SaveJson(saveFileName + ".json");
        nnet.SaveBinary(saveFileName + ".dat");
        Debug.Log("Saved");
    }
    public void LoadSave()
    {
        nnet = new NeuralNetComputeShader(NeuralNet.LoadBinary(saveFileName + ".dat"), layerComputeShader, singlePassComputeShader);
        //nnet = new NeuralNetComputeShader(NeuralNet.LoadJson(saveFileName + ".json"), layerComputeShader, singlePassComputeShader);
    }

    // Start is called before the first frame update
    void Start()
    {

        nnet = new NeuralNetComputeShader(2,1, layerComputeShader, singlePassComputeShader);
        nnet.PopulateLayersRandomly(funcToUse, numNetLayers, minNeuronsPerLayer, maxNeuronsPerLayer);
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
        if (nnet == null) return;
        if (useLog) Debug.Log("Disposing NeuralNetComputeShader (for GPU layer ComputeBuffers)");
        nnet.Dispose();
    }

    void Update()
    {
        if (!thinkCycleInProgress)
            CycleThinkAndLearn();

    }

    int cycleCounter = 0;
    int changeCounter = 0;
    float[] inputs= new float[2];
    float[] errors = new float[1];
    bool thinkInProgress = false;
    bool thinkCycleInProgress=false;

    async void CycleThinkAndLearn()
    {
        thinkCycleInProgress = true;
        while(continuousThink || singleThink)
        {
            if (!thinkInProgress)
            {
                await DoThinkAndLearn();
                if (singleThink)
                {
                    singleThink = false;
                    break;
                }
                if (processUntilCycle != -1 && processUntilCycle >= cycleCounter)
                {
                    continuousThink = false;
                    break;
                }
            }
        }
        thinkCycleInProgress = false;
    }

    async UniTask DoThinkAndLearn()
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
        float[] output = await nnet.GPUThink(inputs);//
        if (useLog)
        {
            await nnet.GetGPUData();
            Debug.Log("Think cycle " + cycleCounter + " done: " + nnet.ToString());
        }
        errors = ComputeErrors(inputs, output);
        await nnet.Backpropagate(errors, learningRate);//  await is test
        if (useLog)
        {
            await nnet.GetGPUData();
            Debug.Log("Backpropagate cycle " + cycleCounter + " done. errors: " + string.Join(",", errors) + nnet.ToString());
        }

        thinkInProgress = false;
    }
    float[] ComputeErrors(float[] input, float[] output)
    {
        //    if (input[2] < 1) //add
        return new float[] { output[0] - (input[0] + input[1]) };  //goal output is (input0 + input1)
                                                                   //   else //subtract 
                                                                   //       return new float[] { output[0] - (input[0] - input[1]) };
    }
}
