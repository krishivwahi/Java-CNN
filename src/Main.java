import data.DataReader;
import data.Image;
import layers.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Main {

    // ⚙ Configuration

    private static final String TRAIN_PATH = "data/mnist_train.csv";
    private static final String TEST_PATH  = "data/mnist_test.csv";
    private static final int MAX_TRAIN   = 3_000;
    private static final int MAX_TEST    = 1_000;
    private static final int EPOCHS      = 1;
    private static final double LR       = 0.01;
    private static final long SEED       = 123L;

    // Network arch

    private static final int NUM_FILTERS  = 8;
    private static final int FILTER_SIZE  = 5;   // 5×5 kernels
    private static final int CONV_STRIDE  = 1;

    private static final int POOL_WINDOW  = 2;
    private static final int POOL_STRIDE  = 2;

    // Derived: conv output = (28-5)/1+1 = 24; pool output = (24-2)/2+1 = 12
    private static final int CONV_OUT = (28 - FILTER_SIZE) / CONV_STRIDE + 1; // 24
    private static final int POOL_OUT = (CONV_OUT - POOL_WINDOW) / POOL_STRIDE + 1; // 12
    private static final int FC_IN    = NUM_FILTERS * POOL_OUT * POOL_OUT; // 1152
    private static final int NUM_CLASSES = 10;

    //  Main FINALLY

    public static void main(String[] args) throws Exception {

        System.out.println("Loading training data …");
        List<Image> trainImages = DataReader.readData(TRAIN_PATH, MAX_TRAIN);
        System.out.println("Loaded " + trainImages.size() + " training images.");

        System.out.println("Loading test data …");
        List<Image> testImages = DataReader.readData(TEST_PATH, MAX_TEST);
        System.out.println("Loaded " + testImages.size() + " test images.\n");

        // Build network

        convolutionLayer    convLayer  = new convolutionLayer(FILTER_SIZE, CONV_STRIDE,
                NUM_FILTERS, 1, 28, 28, SEED, LR);
        maxPoolLayer        poolLayer  = new maxPoolLayer(POOL_STRIDE, POOL_WINDOW,
                NUM_FILTERS, CONV_OUT, CONV_OUT);
        fullyConnectedLayer fcLayer    = new fullyConnectedLayer(FC_IN, NUM_CLASSES, SEED, LR);

        //  Training loop

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            Collections.shuffle(trainImages);

            double totalLoss  = 0;
            int    correct    = 0;

            for (int i = 0; i < trainImages.size(); i++) {
                Image img = trainImages.get(i);

                // Wrap single-channel image in a list for the conv layer
                List<double[][]> input = new ArrayList<>();
                input.add(img.getData());

                // FW pass

                List<double[][]> convOut  = convLayer.convolutionForwardPass(input);
                List<double[][]> poolOut  = poolLayer.maxPoolForwardPass(convOut);
                double[]         fcOut    = fcLayer.fullyConnectedForwardPass(flatten(poolOut));

                //  Loss (cross-entropy)

                int    label    = img.getLabel();
                double loss     = -Math.log(Math.max(fcOut[label], 1e-10));
                totalLoss      += loss;

                if (argmax(fcOut) == label) correct++;

                //  BW pass

                // dL/dOutput = -1/p(correct class), 0 elsewhere
                double[] dLdFcOut = new double[NUM_CLASSES];
                dLdFcOut[label] = -1.0 / Math.max(fcOut[label], 1e-10);

                double[] dLdFcIn   = fcLayer.fullyConnectedBackPass(dLdFcOut);
                List<double[][]> dLdPoolOut = unflatten(dLdFcIn,
                        NUM_FILTERS, POOL_OUT, POOL_OUT);
                List<double[][]> dLdConvOut = poolLayer.maxPoolBackPass(flattenList(dLdPoolOut));

                double[] dLdConvOutFlat = flattenList(dLdConvOut);
                convLayer.convolutionBackPass(dLdConvOutFlat);

                // ── Progress reporting ────────────────────────────────────────
                if ((i + 1) % 1000 == 0) {
                    System.out.printf("  Epoch %d | step %5d/%d | "
                                    + "avg loss: %.4f | train acc: %.2f%%%n",
                            epoch + 1, i + 1, trainImages.size(),
                            totalLoss / (i + 1),
                            100.0 * correct / (i + 1));
                }
            }

            double epochAcc = 100.0 * correct / trainImages.size();
            System.out.printf("Epoch %d complete — avg loss: %.4f | train accuracy: %.2f%%%n%n",
                    epoch + 1, totalLoss / trainImages.size(), epochAcc);
        }

        // Test eval

        System.out.println("Evaluating on test set …");
        int testCorrect = 0;

        for (Image img : testImages) {
            List<double[][]> input = new ArrayList<>();
            input.add(img.getData());

            List<double[][]> convOut = convLayer.convolutionForwardPass(input);
            List<double[][]> poolOut = poolLayer.maxPoolForwardPass(convOut);
            double[]         fcOut   = fcLayer.fullyConnectedForwardPass(flatten(poolOut));

            if (argmax(fcOut) == img.getLabel()) testCorrect++;
        }

        System.out.printf("Test accuracy: %.2f%% (%d / %d)%n",
                100.0 * testCorrect / testImages.size(),
                testCorrect, testImages.size());
    }




    private static double[] flatten(List<double[][]> maps) {
        int total = 0;
        for (double[][] m : maps) total += m.length * m[0].length;
        double[] out = new double[total];
        int idx = 0;
        for (double[][] m : maps) {
            for (double[] row : m) {
                for (double v : row) out[idx++] = v;
            }
        }
        return out;
    }


    private static double[] flattenList(List<double[][]> maps) {
        return flatten(maps);
    }

    /**feature maps  */
    private static List<double[][]> unflatten(double[] flat, int length, int rows, int cols) {
        List<double[][]> out = new ArrayList<>();
        int idx = 0;
        for (int l = 0; l < length; l++) {
            double[][] map = new double[rows][cols];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    map[r][c] = flat[idx++];
                }
            }
            out.add(map);
        }
        return out;
    }

    private static int argmax(double[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }
}
