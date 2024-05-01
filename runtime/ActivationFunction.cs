using System;
using System.Collections.Generic;
using UnityEngine;

namespace EyE.NNET
{
    public enum ActivationFunction
    {
        None,
        ReLU,
        Sigmoid,
        Tanh01
    }
    public static class EnumValues<TEnum> where TEnum : Enum
    {
        public static readonly TEnum[] Values = (TEnum[])Enum.GetValues(typeof(TEnum));
    }

    public static class EnumExtensions
    {
        public static int GetEnumIndex<TEnum>(this TEnum enumValue) where TEnum : Enum
        {
            return Array.IndexOf(EnumValues<TEnum>.Values, enumValue);
        }
        public static TEnum GetRandom<TEnum>() where TEnum : Enum
        {
            return EnumValues<TEnum>.Values[UnityEngine.Random.Range(0,EnumValues<TEnum>.Values.Length)];
        }
    }


    static public class ActivationFunctionExtension
    {
        static public float Activate(this ActivationFunction activationFunction, float value)
        {
            switch (activationFunction)
            {
                case ActivationFunction.None:
                    return value;
                case ActivationFunction.ReLU:
                    return Mathf.Max(0, (float)value);
                case ActivationFunction.Sigmoid:
                    return 1.0f / (1.0f + Mathf.Exp((float)-value));
                case ActivationFunction.Tanh01:
                    return (1.0f + (float)System.Math.Tanh(-value)) * 0.5f;
                default:
                    throw new System.ArgumentException("Unsupported activation function");
            }
        }

        static public float ApplyActivationFunctionDerivative(this ActivationFunction activationFunction, float value)
        {
            switch (activationFunction)
            {
                case ActivationFunction.None:
                    return 1.0f; // Derivative of the identity function is 1
                case ActivationFunction.ReLU:
                    return (value > 0) ? 1.0f : 0.0f; // Derivative of ReLU
                case ActivationFunction.Sigmoid:
                    //float ex = Mathf.Exp(-value);
                    //float f = ex / ((1 + ex) * (1 + ex));
                    //return f;
                    float sigmoid = 1.0f / (1.0f + Mathf.Exp(-value));
                    return sigmoid * (1 - sigmoid); // Derivative of sigmoid
                case ActivationFunction.Tanh01:
                    float tanh = (1.0f + (float)System.Math.Tanh(value * 2.0f - 1.0f)) * 0.5f;
                    return 1 - tanh * tanh; // Derivative of tanh (scaled to [0, 1])
                default:
                    throw new System.ArgumentException("Unsupported activation function");
            }
        }

        static public ActivationFunction RandomActivationFunction()
        {
            return EnumExtensions.GetRandom<ActivationFunction>();
            //int randomIndex = UnityEngine.Random.Range((int)0, System.Enum.GetNames(typeof(ActivationFunction)).Length);
            //return (ActivationFunction)randomIndex;
        }
    }

}