using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Xml;
using System.IO;

namespace Нейронная_сеть_предсказывающая_закрытие_хайпов
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            NeuralNetwork n = new NeuralNetwork();
            //n.RandomizeWeights();
            n.LoadWeights();
            InputLayer i = new InputLayer();
            MessageBox.Show("Проект продержится кругов: " + n.Start(i.Do(Convert.ToDouble(textBox1.Text), Convert.ToDouble(textBox2.Text), Convert.ToDouble(textBox3.Text), Convert.ToDouble(textBox4.Text), Convert.ToDouble(textBox5.Text) )));
        }

        private void button2_Click(object sender, EventArgs e)
        {
            NeuralNetwork n = new NeuralNetwork();
            NeuralNetwork.Train(n);
            MessageBox.Show("Сеть натренирована");
        }
    }
    class InputLayer
    {
        public double[] Do(double i1, double i2, double i3, double i4, double i5)
        {
            return new double[5] { i1, i2, i3, i4, i5 };
        }
    }
    class HiddenNeuron
    {
        public double[] w;
        public HiddenNeuron()
        {
            w = new double[5];
        }
        public void InitWeights(double w1, double w2, double w3, double w4, double w5)
        {
            w[0] = w1; w[1] = w2; w[2] = w3; w[3] = w4; w[4] = w5;
        }
        public double Output(double[] i)
        {
            double sum = 0;
            for(var o = 0; o < i.Length; o++)
            {
                sum += i[o] * w[o];
            }
            return 1 / (1 + Math.Exp(-1 * sum));
        }
    }
    class OutputNeuron
    {
        public double[] w;
        public OutputNeuron()
        {
            w = new double[20];
        }
        public void InitWeights(double w1, double w2, double w3, double w4, double w5, double w6, double w7, double w8, double w9, double w10, double w11, double w12, double w13, double w14, double w15, double w16, double w17, double w18, double w19, double w20)
        {
            w[0] = w1; w[1] = w2; w[2] = w3; w[3] = w4; w[4] = w5;
            w[5] = w6; w[6] = w7; w[7] = w8; w[8] = w9; w[9] = w10;
            w[10] = w11; w[11] = w12; w[12] = w13; w[13] = w14; w[14] = w15;
            w[15] = w16; w[16] = w17; w[17] = w18; w[18] = w19; w[19] = w20;
        }
        public double Output(double[] i)
        {
            double sum = 0;
            for(var j = 0; j < i.Length; j++)
            {
                sum += i[j] * w[j];
            }
            return 1 / (1 + Math.Exp(-1 * sum));
            //return 1 / (1 + Math.Exp(-1 * (i[0] * w[0] + i[1] * w[1] + i[2] * w[2] + i[3] * w[3] + i[4] * w[4] + i[5] * w[5] + i[6] * w[6] + i[7] * w[7] + i[8] * w[8] + i[9] * w[9] + i[10] * w[10] + i[11] * w[11] + i[12] * w[12] + i[13] * w[13] + i[14] * w[14] + i[15] * w[15] + i[16] * w[16] + i[17] * w[17] + i[18] * w[18] + i[19] * w[19])));
        }
    }
    class NeuralNetwork
    {
        public InputLayer input = new InputLayer();
        public HiddenNeuron[] hidden = new HiddenNeuron[20];
        public OutputNeuron[] output = new OutputNeuron[4];
        public NeuralNetwork()
        {
            for (var i = 0; i < 20; i++)
            {
                if (i < 4) output[i] = new OutputNeuron();
                hidden[i] = new HiddenNeuron();
            }
        }
        public void SaveWeights()
        {
            StreamWriter sw = new StreamWriter("memory.dat", false);
            for (var i = 0; i < 20; i++)
            {
                for (var j = 0; j < 5; j++)
                {
                    sw.WriteLine(hidden[i].w[j]);
                }
            }
            for (var i = 0; i < 4; i++)
            {
                for (var j = 0; j < 20; j++)
                {
                    sw.WriteLine(output[i].w[j]);
                }
            }
            sw.Close();
        }
        public void LoadWeights()
        {
            StreamReader sr = new StreamReader("memory.dat");
            for (var i = 0; i < 20; i++)
            {
                for (var j = 0; j < 5; j++)
                {
                    hidden[i].w[j] = Double.Parse(sr.ReadLine());
                }
            }
            for (var i = 0; i < 4; i++)
            {
                for (var j = 0; j < 20; j++)
                {
                    output[i].w[j] = Double.Parse(sr.ReadLine());
                }
            }
            sr.Close();
        }
        public void RandomizeWeights()
        {
            Random r = new Random();
            for (var i = 0; i < 20; i++)
            {
                for(var j = 0; j < 5; j++)
                {
                    hidden[i].w[j] = r.NextDouble();
                }
            }
            for (var i = 0; i < 4; i++)
            {
                for (var j = 0; j < 20; j++)
                {
                    output[i].w[j] = r.NextDouble();
                }
            }
        }
        public double DeltaWeights(double error, double sigmoid)
        {
            return error * sigmoid * (1 - sigmoid);
        }
        double GetMSE(double[] errors)
        {
            double sum = 0;
            for (int i = 0; i < errors.Length; ++i)
                sum += Math.Pow(errors[i], 2);
            return 0.5d * sum;
        }
        double GetCost(double[] mses)
        {
            double sum = 0;
            for (int i = 0; i < mses.Length; ++i)
                sum += mses[i];
            return (sum / mses.Length);
        }
        public int Start(double[] i)
        {
            double[] houtput = new double[20];
            double[] ooutput = new double[4];
            int[] b = new int[4];
            for (var t = 0; t < 20; t++)
            {
                houtput[t] = hidden[t].Output(i);
            }
            for (var y = 0; y < 4; y++)
            {
                ooutput[y] = output[y].Output(houtput);
                b[y] = Convert.ToInt32(Math.Round(ooutput[y]));
            }
            return Convert.ToInt32(b[0].ToString() + b[1].ToString() + b[2].ToString() + b[3].ToString(), 2);
        }
        public static void Train(NeuralNetwork n)
        {
            n.RandomizeWeights();
            //n.LoadWeights();
            InOut data = new InOut();
            int j = 0;
            double[] houtput = new double[20];
            double[] ooutput = new double[4];
            const double threshold = 0.29d;//порог ошибки
            const double learning_rate = 0.01d;
            double[] temp_mses = new double[16];//массив для хранения ошибок итераций
            double temp_cost = 0;//текущее значение ошибки по эпохе
            do
            {
                for (var i = 0; i < 16; i++, j++)
                {
                    double[] oerrors = new double[4];
                    double[] herrors = new double[20];
                    if (data.Refresh(j) == false) j = 0;
                    for (var t = 0; t < 20; t++)
                    {
                        houtput[t] = n.hidden[t].Output(n.input.Do(data.input[0], data.input[1], data.input[2], data.input[3], data.input[4]));
                    }
                    for (var y = 0; y < 4; y++)
                    {
                        ooutput[y] = n.output[y].Output(houtput);
                        oerrors[y] = ooutput[y] - data.output[y];
                        for (var g = 0; g < n.output[y].w.Length; g++)
                        {
                            n.output[y].w[g] -= n.DeltaWeights(oerrors[y], ooutput[y]) * learning_rate * houtput[g];
                            herrors[g] += n.output[y].w[g] * n.DeltaWeights(oerrors[y], ooutput[y]);
                        }
                    }
                    for (var k = 0; k < 20; k++)
                    {
                        for (var l = 0; l < 5; l++)
                        {
                            n.hidden[k].w[l] -= n.DeltaWeights(herrors[k], houtput[k]) * learning_rate * n.input.Do(data.input[0], data.input[1], data.input[2], data.input[3], data.input[4])[l];
                        }
                    }
                    temp_mses[i] = n.GetMSE(oerrors);
                }
                temp_cost = n.GetCost(temp_mses);//вычисление ошибки по эпохе
            } while (temp_cost > threshold);
            n.SaveWeights();
        }
    }
    class InOut
    {
        public double[] input;
        public double[] output;
        public InOut()
        {
            input = new double[5];
            output = new double[4];
        }
        public bool Refresh(int str)
        {
            try
            {
                StreamReader sr = new StreamReader("train.txt");
                for (var j = 0; j < str; j++)
                {
                    sr.ReadLine();
                }
                var s = sr.ReadLine().Split(' ');
                sr.Close();
                input[0] = Double.Parse(s[0]);
                input[1] = Double.Parse(s[1]);
                input[2] = Double.Parse(s[2]);
                input[3] = Double.Parse(s[3]);
                input[4] = Double.Parse(s[4]);
                output[0] = Double.Parse(s[5]);
                output[1] = Double.Parse(s[6]);
                output[2] = Double.Parse(s[7]);
                output[3] = Double.Parse(s[8]);
                return true;
            }
            catch { return false; }
        }
    }
}