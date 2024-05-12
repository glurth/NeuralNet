using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using EyE.NNET;
using UnityEditor;
using TMPro;

public class Visualizer : MonoBehaviour
{
    public NeuralNet neuralNetwork = null; //this value must be assigned for this class to do anything
    public GameObject neuronPrefab; // Prefab for neuron (sphere)
    public GameObject connectionPrefab; // Prefab for connection (cylinder)
    public TMP_Text textPrefab; // Prefab for displaying text
    

    public Material positiveConnectionWeight;
    public Material negativeConnectionWeight;
    public Material positiveBiasWeight;
    public Material negativeBiasWeight;

    public float neuronAvgScale = 50f;
    public float weightAvgScale = 50f;

    public Color neuronColor = Color.blue;
    public Color connectionColor = Color.white;
    public bool writeValuesNotBias = false;

    public bool useLog = true;

    private List<List<NeuronDisplayObjects>> neurons = new List<List<NeuronDisplayObjects>>();
    

    struct NeuronDisplayObjects
    {
        public GameObject neuronSphere;
        public TMP_Text biasText;

        public NeuronDisplayObjects(GameObject neuronSphere, TMP_Text biasText)
        {
            this.neuronSphere = neuronSphere;
            this.biasText = biasText;
        }
    }

    struct LayerConnectionObjects
    {
        public GameObject cylinder;
        public TMP_Text weightText;

        public LayerConnectionObjects(GameObject cylinder, TMP_Text weightText)
        {
            this.cylinder = cylinder;
            this.weightText = weightText;
        }
    }
    private List<List<List<LayerConnectionObjects>>> connections = new List<List<List<LayerConnectionObjects>>>();

    public class WeightedRunningAverage
    {
        private float weightedSum;
        private float totalWeight;

        // Adjust this factor to control the weighting towards more recent results
        private float decayFactor = 0.9f;

        public float Average { get { return totalWeight > 0 ? weightedSum / totalWeight : 0f; } }

        public void AddValue(float value)
        {
            weightedSum = value + decayFactor * weightedSum;
            totalWeight = 1 + decayFactor * totalWeight;
        }

        public void Reset()
        {
            weightedSum = 0;
            totalWeight = 0;
        }
    }
    List<WeightedRunningAverage> errors = new List<WeightedRunningAverage>();
    private void Start()
    {
     
    }

    bool creationCompleted = false;

    private void CreateNeuralNetworkObjects()
    {
        if (neuralNetwork == null)
            return;
        if (creationCompleted) return;
        creationCompleted = true;

        List<NeuronDisplayObjects> inputNeurons = new List<NeuronDisplayObjects>();
        neurons.Add(inputNeurons);
        for (int i = 0; i < neuralNetwork.NumInputs; i++)
        {
            Vector3 position = GetNeuronPosition(-1, i);
            Vector3 worldPosition = transform.TransformPoint(position);
            GameObject neuron = Instantiate(neuronPrefab, worldPosition, Quaternion.identity, transform);
            TMP_Text text = Instantiate(textPrefab, worldPosition + Vector3.back * 0.5f, Quaternion.identity, transform);
            text.text = "0";
            inputNeurons.Add(new NeuronDisplayObjects(neuron, text));
        }

        Quaternion topToFront = Quaternion.FromToRotation(Vector3.up, Vector3.forward);
        for (int i = 0; i < neuralNetwork.Layers.Count; i++)
        {
            NetLayer layer = neuralNetwork.Layers[i];
            List<NeuronDisplayObjects> layerNeurons = new List<NeuronDisplayObjects>();
            neurons.Add(layerNeurons);//stored at index i+1

            List<List<LayerConnectionObjects>> layerConnections = new List<List<LayerConnectionObjects>>();
            connections.Add(layerConnections); //stored at index i

            for (int j = 0; j < layer.NumNeurons; j++)
            {
                Vector3 position = GetNeuronPosition(i, j);
                Vector3 worldPosition = transform.TransformPoint(position);
                GameObject neuron = Instantiate(neuronPrefab, worldPosition, Quaternion.identity, transform);
                TMP_Text text = Instantiate(textPrefab, worldPosition + Vector3.back * 0.5f, Quaternion.identity, transform);
                text.text = "0";
                layerNeurons.Add(new NeuronDisplayObjects(neuron, text));

                //if (i > 0)//if previous layer exists
                {
                    //List<List<LayerConnectionObjects>> layerConnections = connections[i];// - 1];
                    List<LayerConnectionObjects> neuronConnections = new List<LayerConnectionObjects>();
                    layerConnections.Add(neuronConnections);//has index of [j]
                    int numInputs;
                    if (i > 0) numInputs = neuralNetwork.Layers[i - 1].NumNeurons;
                    else numInputs = neuralNetwork.NumInputs;
                    for (int k = 0; k < numInputs; k++)
                    {
                        Vector3 endPos = GetNeuronPosition(i - 1, k);

                        Vector3 offset = endPos - position;
                        Quaternion orient = Quaternion.LookRotation(offset) * topToFront;
                        Vector3 midPoint = (endPos + position) / 2f;
                        midPoint = transform.TransformPoint(midPoint);
                        GameObject connection = Instantiate(connectionPrefab, midPoint, orient, transform);
                        connection.transform.localScale = new Vector3(0.1f, offset.magnitude / 2, 0.1f);

                        TMP_Text weightText = Instantiate(textPrefab, midPoint - offset * 0.25f + Vector3.back * 0.2f, Quaternion.identity, transform);
                        weightText.text = "0";
                        LayerConnectionObjects conection = new LayerConnectionObjects(connection, weightText);
                        neuronConnections.Add(conection);//has index of [k]
                    }
                }
            }
        }
        for (int i = 0; i < neuralNetwork.NumOutputs; i++)
        {
            errors.Add(new WeightedRunningAverage());
        } 
    }


    private void UpdateNeuralNetworkObjects()
    {
        if (neuralNetwork == null || neuralNetwork.lastInputs==null || neuralNetwork.Layers[0].lastOutputs==null)
            return;
        if (useLog)
        {
            if (neuralNetwork.lastOutputErrors != null)
            {
                for (int i = 0; i < neuralNetwork.NumOutputs; i++)
                {
                    errors[i].AddValue(neuralNetwork.lastOutputErrors[i]);
                    if(useLog)
                        Debug.Log(name + " Running Avg Error: " + errors[i].Average);
                }
            }
        }
        for (int i = 0; i < neuralNetwork.NumInputs; i++)
        {
            float bias = 1;
            NeuronDisplayObjects neuron = neurons[0][i];
            float diameter = ZeroInfToZeroOne(bias/ neuronAvgScale, 2) + 0.6f;
            neuron.neuronSphere.transform.localScale = Vector3.one * (diameter);
            neuron.neuronSphere.GetComponent<MeshRenderer>().sharedMaterial = bias>0 ? positiveBiasWeight: negativeBiasWeight;
            if(writeValuesNotBias)
                neuron.biasText.text = Scientific3(neuralNetwork.lastInputs[i], 3);
            else
                neuron.biasText.text = Scientific3(bias, 3);
          //  Vector3 pos = neuron.biasText.transform.localPosition;
          //  pos.z = -diameter / 2f;
          //  neuron.biasText.transform.localPosition = pos;
            neuron.biasText.transform.localPosition = neuron.neuronSphere.transform.localPosition;
            neuron.biasText.transform.LookAt(Camera.main.transform.position);//.localRotation;
            neuron.biasText.transform.Rotate(0, 180, 0);
            neuron.biasText.transform.position -= Camera.main.transform.forward * diameter;// / 2f;
        }
        Quaternion topToFront = Quaternion.FromToRotation(Vector3.up, Vector3.forward);
        for (int i = 0; i < neuralNetwork.Layers.Count; i++)
        {
            NetLayer layer = neuralNetwork.Layers[i];
            if (layer.lastOutputs == null)
                return;
            for (int j = 0; j < layer.NumNeurons; j++)
            {

                float bias = layer.biases[j];
                NeuronDisplayObjects neuron = neurons[i+1][j];
                float diameter = ZeroInfToZeroOne(bias, neuronAvgScale) + 0.6f;
                neuron.neuronSphere.transform.localScale = Vector3.one * (diameter);
                Vector3 position = GetNeuronPosition(i, j);
                neuron.neuronSphere.transform.localPosition = position;
                neuron.neuronSphere.GetComponent<MeshRenderer>().sharedMaterial = bias >= 0 ? positiveBiasWeight : negativeBiasWeight;
                if (writeValuesNotBias)
                    neuron.biasText.text = Scientific3(layer.lastOutputs[j], 3);
                else
                    neuron.biasText.text = Scientific3(bias, 3);
                
                //Vector3 pos = position + Vector3.back * diameter / 2f;
                Vector3 pos = position + -Camera.main.transform.forward * diameter / 2f;
                neuron.biasText.transform.localPosition = position;
                
                neuron.biasText.transform.LookAt(Camera.main.transform.position);//.localRotation;
                neuron.biasText.transform.Rotate(0, 180, 0);
                neuron.biasText.transform.position -= Camera.main.transform.forward * diameter;// / 2f;
                int numInputs;
                if (i > 0) numInputs = neuralNetwork.Layers[i - 1].NumNeurons;
                else numInputs = neuralNetwork.NumInputs;
                for (int k = 0; k < numInputs; k++)
                {
                    LayerConnectionObjects connection = connections[i][j][k];
                    float weight = layer.weights[j, k];
                    float size = ZeroInfToZeroOne(weight,weightAvgScale) + 0.1f;
                    Vector3 endPos = GetNeuronPosition(i - 1, k);
                    Vector3 offset = endPos - position;
                    Quaternion orient = Quaternion.LookRotation(offset) * topToFront;
                    Vector3 midPoint = (endPos + position) / 2f;
                    float length = offset.magnitude / 2;
                    connection.cylinder.transform.localPosition = midPoint;
                    connection.cylinder.transform.localRotation = orient;
                    connection.cylinder.transform.localScale = new Vector3(size, length, size);
                    connection.cylinder.GetComponent<MeshRenderer>().sharedMaterial = weight >= 0 ? positiveConnectionWeight : negativeConnectionWeight;
                    connection.weightText.text = Scientific3(weight, 3);//  weight.ToString("E2");
                   // Vector3 textPos = midPoint - Camera.main.transform.forward * size / 2f;
                    connection.weightText.transform.localPosition = midPoint;
                    connection.weightText.transform.position -= Camera.main.transform.forward * size / 2f;
                    connection.weightText.transform.LookAt(Camera.main.transform.position);
                    connection.weightText.transform.Rotate(0, 180, 0);
                }
            }
        }
    }

    public float updateTime;
    float lastUpdateTime = 0;
    void Update()
    {
        if (lastUpdateTime + updateTime < Time.time)
        {
            CreateNeuralNetworkObjects();
            UpdateNeuralNetworkObjects();
            lastUpdateTime = Time.time;
        }
    }


    private Vector3 GetNeuronPosition(int layer, int neuronNumber)
    {
        float mid;
        float r = 0;
        if (layer < 0)
        {
            mid = neuralNetwork.NumInputs;
          //  r = 3;
        }
        else
        {
            mid = neuralNetwork.Layers[layer].NumNeurons;
        //    r = 3f;// * ZeroInfToZeroOne(neuralNetwork.Layers[layer].biases[neuronNumber], 1);
        }
        
        r = (mid - 1) * 1f;
        float neuronFrac = neuronNumber / mid;
        float x = r*Mathf.Sin(Mathf.PI * 2 * neuronFrac);
        float z = r*Mathf.Cos(Mathf.PI * 2 * neuronFrac);
        // float x = neuronNumber * 2f;
        //x= x - mid
        float y = layer * 2f;
        
        
        return new Vector3(x, y, z) * 3;
    }

    //a defines what value of X will output 0.5f
    static float ZeroInfToZeroOne(float x, float a=1)
    {
        if (a == 0) throw new System.ArgumentException("Cannot pass zero to ZeroInfToZeroOne(x,a) for parameter a");
        x /= a;
        if (x < 0) x = -x;
        return 1 + 1 / (-x - 1);


        //float val = (ActivationFunction.Sigmoid.Activate(Mathf.Abs(x))-0.5f) *2f;
        //return  val * a;

        /*
        float scaleFactor = -1f / a;
        float y = 1f - Mathf.Exp(scaleFactor * x);
        y = Mathf.Clamp01(y);
        return y;*/
    }
    /*static string Scientific3(double value, int precision)
    {
        string fstr1 = "{0:E" + (precision + 2).ToString() + "}";
        string step1 = string.Format(fstr1, value);
        int index1 = step1.ToLower().IndexOf('e');
        if (index1 < 0 || index1 == step1.Length - 1)
            throw new System.Exception();
        decimal coeff = System.Convert.ToDecimal(step1.Substring(0, index1));
        int index = System.Convert.ToInt32(step1.Substring(index1 + 1));

        while (index % 3 != 0)
        {
            index--;
            coeff *= 10;
        }
        if (index == 0)
            return coeff.ToString("0.00");
        string fstr2 = "{0:G" + precision.ToString() + "}e{1}{2:D}";
        return string.Format(fstr2, coeff, ((index < 0) ? "-" : "+"), Mathf.Abs(index));
    }*/
    /// <summary>
    /// Converts a floating-point number to a string with a specified number of significant digits 
    /// and exponent in a scientific notation format.
    /// </summary>
    /// <param name="number">The floating-point number to convert.</param>
    /// <param name="significantDigits">The desired number of significant digits 
    /// (including leading zeros but excluding trailing zeros).</param>
    /// <returns>The formatted string representation of the number.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if significantDigits is less than 1.</exception>
    public static string Scientific3(float number, int significantDigits=3)
    {
        if (significantDigits < 1)
        {
            throw new System.ArgumentOutOfRangeException(nameof(significantDigits), "significantDigits must be at least 1");
        }

        // Handle special cases (infinity, NaN)
        if (float.IsInfinity(number))
        {
            return number > 0 ? "+INF" : "-INF";
        }
        else if (float.IsNaN(number))
        {
            return "NaN";
        }
       
        int exponent = 0;
        float sign = Mathf.Sign(number);
        number = Mathf.Abs(number);
        while (true)
        {
            if (number == 0) return "0";
            if (number > 100f) //scale down number;
            {
                exponent += 3;
                number /= 100f;
            }
            else if (number < .1f)
            {
                exponent -= 3;
                number *= 100f;
            }
            else
                break;
        }
        string output="";
        if (sign < 0)
            output = "-";
        output += number.ToString("F" + significantDigits);
        if (exponent != 0)
            output += "E" + exponent.ToString();
        return output;

        /*
        // Extract integer and fractional parts (absolute value)
        int integerPart = (int)Mathf.Floor(Mathf.Abs(number));
        float fractionalPart = Mathf.Abs(number) - integerPart;

        // Find exponent (multiple of 3) to achieve significantDigits
        int exponent = 0;
        while (significantDigits > 0 && (integerPart > 0 || fractionalPart > 0))
        {
            if (integerPart > 0)
            {
                integerPart /= 10;
            }
            else
            {
                fractionalPart *= 10;
            }
            exponent -= 3;
            significantDigits--;
        }

        // Format number string
        string formattedNumber = (number >= 0 ? "" : "-") + (integerPart > 0 ? integerPart.ToString() : "0");
        if (significantDigits > 0)
        {
            float pow = Mathf.Pow(10, significantDigits);
            float scaledFractionalPart = fractionalPart * pow;
            int roundedPart = Mathf.RoundToInt(scaledFractionalPart);
            fractionalPart = roundedPart / pow;
            formattedNumber += "." + fractionalPart.ToString("F" + significantDigits);
        }

        // Add exponent if needed
        return exponent != 0 ? formattedNumber + "E" + exponent : formattedNumber;*/
    }

}
