package data;

public class Image {

    private double[][] data;
    private int label;

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Label: ").append(label).append("\n");

        for (double[] row : data) {
            for (double pixel : row) {
                // showing you in ASCII, visuals basically
                sb.append(pixel > 0.5 ? "# " : ". ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
