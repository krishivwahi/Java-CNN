package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class DataReader {

    private static final int IMAGE_SIZE = 28;

    public static List<Image> readData(String path, int maxImages) throws IOException {
        List<Image> images = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line = br.readLine();

            int count = 0;
            while ((line = br.readLine()) != null) {
                if (maxImages >= 0 && count >= maxImages) break;

                String[] tokens = line.split(",");
                if (tokens.length != IMAGE_SIZE * IMAGE_SIZE + 1) continue;

                int label = Integer.parseInt(tokens[0].trim());
                double[][] data = new double[IMAGE_SIZE][IMAGE_SIZE];

                for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
                    int row = i / IMAGE_SIZE;
                    int col = i % IMAGE_SIZE;
                    data[row][col] = Integer.parseInt(tokens[i + 1].trim()) / 255.0;
                }

                images.add(new Image(data, label));
                count++;
            }
        }
        return images;
    }
}