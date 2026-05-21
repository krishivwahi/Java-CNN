package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class convolutionLayer extends layer {

    private final int stepSize;
    private final int filterSize;
    private final int numFilters;
    private final int inLength;
    private final int inRows;
    private final int inCols;
    private final double learningRate;

    private List<double[][][]> filters;
    private double[] bias;
    private List<double[][]> lastInput;

    public convolutionLayer(int filterSize, int stepSize, int numFilters,
                            int inLength, int inRows, int inCols,
                            long seed, double learningRate) {
        this.filterSize   = filterSize;
        this.stepSize     = stepSize;
        this.numFilters   = numFilters;
        this.inLength     = inLength;
        this.inRows       = inRows;
        this.inCols       = inCols;
        this.learningRate = learningRate;
        generateRandomFilters(seed);
    }

    private void generateRandomFilters(long seed) {
        filters = new ArrayList<>();
        bias    = new double[numFilters];
        Random  rng   = new Random(seed);
        double  scale = Math.sqrt(2.0 / (inLength * filterSize * filterSize));

        for (int f = 0; f < numFilters; f++) {
            double[][][] fs = new double[inLength][filterSize][filterSize];
            for (int l = 0; l < inLength; l++)
                for (int r = 0; r < filterSize; r++)
                    for (int c = 0; c < filterSize; c++)
                        fs[l][r][c] = rng.nextGaussian() * scale;
            filters.add(fs);
        }
    }

    // FW pass

    public List<double[][]> convolutionForwardPass(List<double[][]> input) {
        lastInput = input;
        List<double[][]> output = new ArrayList<>();

        for (int f = 0; f < numFilters; f++) {
            double[][] sum = new double[getOutputRows()][getOutputCols()];
            for (int l = 0; l < inLength; l++) {
                double[][] cr = convolve(input.get(l), filters.get(f)[l]);
                for (int r = 0; r < getOutputRows(); r++)
                    for (int c = 0; c < getOutputCols(); c++)
                        sum[r][c] += cr[r][c];
            }
            for (int r = 0; r < getOutputRows(); r++)
                for (int c = 0; c < getOutputCols(); c++)
                    sum[r][c] = relu(sum[r][c] + bias[f]);
            output.add(sum);
        }
        return output;
    }

    private double[][] convolve(double[][] input, double[][] filter) {
        double[][] out = new double[getOutputRows()][getOutputCols()];
        for (int r = 0; r < getOutputRows(); r++)
            for (int c = 0; c < getOutputCols(); c++) {
                double s = 0;
                for (int fr = 0; fr < filterSize; fr++)
                    for (int fc = 0; fc < filterSize; fc++)
                        s += input[r * stepSize + fr][c * stepSize + fc] * filter[fr][fc];
                out[r][c] = s;
            }
        return out;
    }

    private double relu(double x)           { return Math.max(0, x); }
    private double reluDeriv(double x)      { return x > 0 ? 1.0 : 0.0; }

    // BW pass

    public double[] convolutionBackPass(double[] dLdOFlat) {
        int outRows = getOutputRows();
        int outCols = getOutputCols();

        double[][][] dLdI3D = new double[inLength][inRows][inCols];

        for (int f = 0; f < numFilters; f++) {
            double[][] preAct = preActivation(f);
            double[][] gated  = new double[outRows][outCols];
            for (int r = 0; r < outRows; r++)
                for (int c = 0; c < outCols; c++) {
                    int i = f * outRows * outCols + r * outCols + c;
                    gated[r][c] = dLdOFlat[i] * reluDeriv(preAct[r][c] + bias[f]);
                }

            // Bias gradient
            double bGrad = 0;
            for (int r = 0; r < outRows; r++)
                for (int c = 0; c < outCols; c++) bGrad += gated[r][c];
            bias[f] -= learningRate * bGrad;

            for (int l = 0; l < inLength; l++) {
                // Filter gradient
                double[][] filterGrad = inputCrossCorrelate(lastInput.get(l), gated);
                for (int fr = 0; fr < filterSize; fr++)
                    for (int fc = 0; fc < filterSize; fc++)
                        filters.get(f)[l][fr][fc] -= learningRate * filterGrad[fr][fc];

                // Input gradient
                double[][] dLdI_part = fullConvolve(gated, filters.get(f)[l]);
                for (int r = 0; r < inRows; r++)
                    for (int c = 0; c < inCols; c++)
                        dLdI3D[l][r][c] += dLdI_part[r][c];
            }
        }

        double[] flat = new double[inLength * inRows * inCols];
        int idx = 0;
        for (int l = 0; l < inLength; l++)
            for (int r = 0; r < inRows; r++)
                for (int c = 0; c < inCols; c++)
                    flat[idx++] = dLdI3D[l][r][c];
        return flat;
    }

    private double[][] preActivation(int f) {
        double[][] sum = new double[getOutputRows()][getOutputCols()];
        for (int l = 0; l < inLength; l++) {
            double[][] cr = convolve(lastInput.get(l), filters.get(f)[l]);
            for (int r = 0; r < getOutputRows(); r++)
                for (int c = 0; c < getOutputCols(); c++)
                    sum[r][c] += cr[r][c];
        }
        return sum;
    }

    private double[][] inputCrossCorrelate(double[][] input, double[][] kernel) {
        int kRows = kernel.length, kCols = kernel[0].length;
        double[][] result = new double[filterSize][filterSize];
        for (int fr = 0; fr < filterSize; fr++)
            for (int fc = 0; fc < filterSize; fc++) {
                double s = 0;
                for (int r = 0; r < kRows; r++)
                    for (int c = 0; c < kCols; c++)
                        s += input[fr + r * stepSize][fc + c * stepSize] * kernel[r][c];
                result[fr][fc] = s;
            }
        return result;
    }

    private double[][] fullConvolve(double[][] input, double[][] filter) {
        double[][] rot = rotate180(filter);
        int pad = filterSize - 1;
        double[][] padded = new double[input.length + 2*pad][input[0].length + 2*pad];
        for (int r = 0; r < input.length; r++)
            for (int c = 0; c < input[0].length; c++)
                padded[r + pad][c + pad] = input[r][c];

        double[][] result = new double[inRows][inCols];
        for (int r = 0; r < inRows; r++)
            for (int c = 0; c < inCols; c++) {
                double s = 0;
                for (int fr = 0; fr < filterSize; fr++)
                    for (int fc = 0; fc < filterSize; fc++)
                        s += padded[r + fr][c + fc] * rot[fr][fc];
                result[r][c] = s;
            }
        return result;
    }

    private double[][] rotate180(double[][] m) {
        int rows = m.length, cols = m[0].length;
        double[][] r = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r[i][j] = m[rows-1-i][cols-1-j];
        return r;
    }

    //layer overrides :)

    @Override
    public double[] getOutput(List<double[][]> input) {
        return vectorize(convolutionForwardPass(input));
    }

    @Override
    public double[] getOutput(double[] input) { return new double[0]; }

    @Override
    public void backPropogation(double[] dLdO) { convolutionBackPass(dLdO); }

    @Override
    public void backPropogation(List<Double>[][] dLdO) {
        double[] flat = new double[getOutputElements()];
        int idx = 0;
        for (List<Double>[] row : dLdO)
            for (List<Double> slice : row)
                for (double v : slice) flat[idx++] = v;
        backPropogation(flat);
    }
    @Override public int getOutputLength()   { return numFilters; }
    @Override public int getOutputRows()     { return (inRows - filterSize) / stepSize + 1; }
    @Override public int getOutputCols()     { return (inCols - filterSize) / stepSize + 1; }
    @Override public int getOutputElements() { return getOutputLength() * getOutputRows() * getOutputCols(); }

    private double[] vectorize(List<double[][]> maps) {
        double[] out = new double[getOutputElements()];
        int idx = 0;
        for (double[][] map : maps)
            for (double[] row : map)
                for (double v : row) out[idx++] = v;
        return out;
    }
}
