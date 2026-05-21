package layers;

import java.util.List;
import java.util.Random;

public class fullyConnectedLayer extends layer {

    private final int    inLength;
    private final int    outLength;
    private final double learningRate;

    private double[][] weights;
    private double[]   biases;
    private double[]   lastZ;
    private double[]   lastInput;

    public fullyConnectedLayer(int inLength, int outLength, long seed, double learningRate) {
        this.inLength     = inLength;
        this.outLength    = outLength;
        this.learningRate = learningRate;
        generateRandomWeights(seed);
    }

    private void generateRandomWeights(long seed) {
        weights = new double[outLength][inLength];
        biases  = new double[outLength];
        Random rng   = new Random(seed);
        double scale = Math.sqrt(2.0 / (inLength + outLength));
        for (int o = 0; o < outLength; o++)
            for (int i = 0; i < inLength; i++)
                weights[o][i] = rng.nextGaussian() * scale;
    }

    public double[] fullyConnectedForwardPass(double[] input) {
        lastInput = input;
        lastZ     = new double[outLength];
        for (int o = 0; o < outLength; o++) {
            double s = biases[o];
            for (int i = 0; i < inLength; i++) s += weights[o][i] * input[i];
            lastZ[o] = s;
        }
        return softmax(lastZ);
    }

    private double[] softmax(double[] z) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : z) max = Math.max(max, v);
        double[] exp = new double[z.length];
        double   sum = 0;
        for (int i = 0; i < z.length; i++) { exp[i] = Math.exp(z[i] - max); sum += exp[i]; }
        for (int i = 0; i < z.length; i++) exp[i] /= sum;
        return exp;
    }

    public double[] fullyConnectedBackPass(double[] dLdO) {
        double[] sm   = softmax(lastZ);
        double[] dLdZ = new double[outLength];
        for (int i = 0; i < outLength; i++)
            for (int j = 0; j < outLength; j++) {
                double jac = (i == j) ? sm[i] * (1 - sm[i]) : -sm[i] * sm[j];
                dLdZ[i] += dLdO[j] * jac;
            }

        double[] dLdI = new double[inLength];
        for (int o = 0; o < outLength; o++) {
            biases[o] -= learningRate * dLdZ[o];
            for (int i = 0; i < inLength; i++) {
                dLdI[i]       += dLdZ[o] * weights[o][i];
                weights[o][i] -= learningRate * dLdZ[o] * lastInput[i];
            }
        }
        return dLdI;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        return getOutput(matrixToVector(input));
    }

    @Override
    public double[] getOutput(double[] input) { return fullyConnectedForwardPass(input); }

    @Override
    public void backPropogation(double[] dLdO) { fullyConnectedBackPass(dLdO); }

    @Override
    public void backPropogation(List<Double>[][] dLdO) {
        double[] flat = new double[outLength];
        int idx = 0;
        for (List<Double>[] row : dLdO)
            for (List<Double> slice : row)
                for (double v : slice) flat[idx++] = v;
        backPropogation(flat);
    }

    @Override public int getOutputLength()   { return 1; }
    @Override public int getOutputRows()     { return 1; }
    @Override public int getOutputCols()     { return outLength; }
    @Override public int getOutputElements() { return outLength; }
}