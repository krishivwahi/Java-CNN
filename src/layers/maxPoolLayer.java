package layers;

import java.util.ArrayList;
import java.util.List;

public class maxPoolLayer extends layer {

    private int stepSize;
    private int windowSize;
    private int inLength;
    private int inRows;
    private int inClns;

    List<double[][]> lastInput;

    public maxPoolLayer(int stepSize, int windowSize, int inLength, int inRows, int inClns) {
        this.stepSize   = stepSize;
        this.windowSize = windowSize;
        this.inLength   = inLength;
        this.inRows     = inRows;
        this.inClns     = inClns;
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {
        List<double[][]> output = new ArrayList<>();
        lastInput = input;
        for (int l = 0; l < input.size(); l++)
            output.add(pool(input.get(l)));
        return output;
    }

    public double[][] pool(double[][] input) {
        double[][] output = new double[getOutputRows()][getOutputCols()];
        for (int r = 0; r < getOutputRows(); r++)
            for (int c = 0; c < getOutputCols(); c++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int x = 0; x < windowSize; x++)
                    for (int y = 0; y < windowSize; y++)
                        if (input[r * stepSize + x][c * stepSize + y] > max)
                            max = input[r * stepSize + x][c * stepSize + y];
                output[r][c] = max;
            }
        return output;
    }

    public List<double[][]> maxPoolBackPass(double[] dLdOFlat) {
        List<double[][]> dLdO = vectorToMatrix(dLdOFlat, inLength, getOutputRows(), getOutputCols());
        List<double[][]> dLdI = new ArrayList<>();
        for (int l = 0; l < inLength; l++)
            dLdI.add(maxPoolBackPassSingle(dLdO.get(l), lastInput.get(l)));
        return dLdI;
    }

    private double[][] maxPoolBackPassSingle(double[][] dLdO, double[][] input) {
        double[][] dLdI = new double[inRows][inClns];
        for (int r = 0; r < getOutputRows(); r++)
            for (int c = 0; c < getOutputCols(); c++) {
                double max  = Double.NEGATIVE_INFINITY;
                int    maxR = r * stepSize;
                int    maxC = c * stepSize;
                for (int x = 0; x < windowSize; x++)
                    for (int y = 0; y < windowSize; y++) {
                        int curR = r * stepSize + x;
                        int curC = c * stepSize + y;
                        if (input[curR][curC] > max) {
                            max  = input[curR][curC];
                            maxR = curR;
                            maxC = curC;
                        }
                    }
                dLdI[maxR][maxC] += dLdO[r][c];
            }
        return dLdI;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        return matrixToVector(maxPoolForwardPass(input));
    }

    @Override
    public double[] getOutput(double[] input) { return new double[0]; }

    @Override
    public void backPropogation(double[] dLdO) { maxPoolBackPass(dLdO); }

    @Override
    public void backPropogation(List<Double>[][] dLdO) {
        int outRows = getOutputRows();
        int outCols = getOutputCols();
        List<double[][]> converted = new ArrayList<>();
        for (List<Double>[] row : dLdO) {
            double[][] slice = new double[outRows][outCols];
            for (int r = 0; r < outRows; r++)
                for (int c = 0; c < outCols; c++)
                    slice[r][c] = row[r].get(c);
            converted.add(slice);
        }
        backPropogation(matrixToVector(converted));
    }

    @Override public int getOutputLength()   { return inLength; }
    @Override public int getOutputRows()     { return (inRows  - windowSize) / stepSize + 1; }
    @Override public int getOutputCols()     { return (inClns  - windowSize) / stepSize + 1; }
    @Override public int getOutputElements() { return getOutputLength() * getOutputRows() * getOutputCols(); }
}