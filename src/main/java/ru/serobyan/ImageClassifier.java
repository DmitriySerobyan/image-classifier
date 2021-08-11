package ru.serobyan;

import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Random;

public class ImageClassifier {

    private static final String RESOURCES_FOLDER_PATH = "C:\\Users\\d\\Downloads\\mnist_png";

    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;

    private static final int N_SAMPLES_TRAINING = 60000;
    private static final int N_SAMPLES_TESTING = 10000;

    private static final int N_OUTCOMES = 10;
    private static long start;

    public static void main(String[] args) throws IOException {
        start = System.currentTimeMillis();
        DataSetIterator dataSetIterator = getDataSetIterator(RESOURCES_FOLDER_PATH + "/training", N_SAMPLES_TRAINING);
        buildModel(dataSetIterator);
    }

    private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
        var folder = new File(folderPath);
        var subFolders = folder.listFiles();

        var nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH);
        var scaler = new ImagePreProcessingScaler(0, 1);
        var rit = new ResizeImageTransform(WIDTH, HEIGHT);

        var input = Nd4j.create(nSamples, HEIGHT * WIDTH);
        var output = Nd4j.create(nSamples, N_OUTCOMES);

        int n = 0;
        for (var digitFolder : subFolders) {
            var labelDigit = Integer.parseInt(digitFolder.getName());
            var imageFiles = digitFolder.listFiles();

            for (var imgFile : imageFiles) {
                var writableImg = nativeImageLoader.asWritable(imgFile);
                writableImg = rit.transform(writableImg);
                var img = nativeImageLoader.asRowVector(writableImg.getFrame());
                scaler.transform(img);
                input.putRow(n, img);
                output.put(n, labelDigit, 1.0);
                n++;
            }
        }
        var dataSet = new DataSet(input, output);
        var listDataSet = dataSet.asList();
        Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
        return new ListDataSetIterator<>(listDataSet, 10);
    }

    private static void buildModel(DataSetIterator dsi) throws IOException {
        System.out.println("Build Model...");
        var conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Nesterovs(0.006, 0.9))
            .l2(1e-4).list()
            .layer(
                new DenseLayer.Builder()
                    .nIn(HEIGHT * WIDTH)
                    .nOut(1000)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER).build()
            )
            .layer(
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(1000)
                    .nOut(N_OUTCOMES)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build()
            )
            .build();

        var model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(500));

        System.out.println("Train Model...");
        model.fit(dsi);

        System.out.println("Evaluating Model...");
        var testDsi = getDataSetIterator(RESOURCES_FOLDER_PATH + "/testing", N_SAMPLES_TESTING);
        var eval = model.evaluate(testDsi);
        System.out.println(eval.stats());

        var end = System.currentTimeMillis();
        var totalTime = (double) (end - start) / 1000.0;
        System.out.println("Total time: " + totalTime + " seconds");
    }

}