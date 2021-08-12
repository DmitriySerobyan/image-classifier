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
import java.util.*;

public class ImageClassifier {

    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;

    private final File[] trainingImageFolders;
    private final File[] testingImageFolders;
    private final int nOutcomes;
    private final Map<String, Integer> labelToLabelNumber;
    private MultiLayerNetwork model;

    public ImageClassifier(File[] trainingImageFolders, File[] testingImageFolders, int nOutcomes, Map<String, Integer> labelToLabelNumber) {
        this.trainingImageFolders = trainingImageFolders;
        this.testingImageFolders = testingImageFolders;
        this.nOutcomes = nOutcomes;
        this.labelToLabelNumber = labelToLabelNumber;
    }

    static ImageClassifier of(String trainingImageFoldersPath, String testingImageFoldersPath) {
        var trainingImageFolder = new File(trainingImageFoldersPath);
        var testingImageFolder = new File(testingImageFoldersPath);
        var trainingImageFolders = trainingImageFolder.listFiles();
        var testingImageFolders = testingImageFolder.listFiles();
        var nOutcomes = trainingImageFolders.length;
        var labelToLabelNumber = new HashMap<String, Integer>();
        var labelNumber = 0;
        for (var folder : trainingImageFolders) {
            var label = folder.getName();
            labelToLabelNumber.put(label, labelNumber);
            labelNumber++;
        }
        return new ImageClassifier(trainingImageFolders, testingImageFolders, nOutcomes, labelToLabelNumber);
    }

    public void buildModel() {
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
                    .nOut(nOutcomes)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build()
            )
            .build();

        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(500));
    }

    public DataSetIterator getTrainingDataSetIterator() throws IOException {
        return getDataSetIterator(trainingImageFolders);
    }

    public DataSetIterator getTestingDataSetIterator() throws IOException {
        return getDataSetIterator(testingImageFolders);
    }

    private DataSetIterator getDataSetIterator(File[] folders) throws IOException {
        var nSamples = Arrays.stream(folders)
            .mapToInt(subFolder -> subFolder.listFiles().length)
            .sum();

        var nativeImageLoader = new NativeImageLoader();
        var scaler = new ImagePreProcessingScaler(0, 1);
        var resizer = new ResizeImageTransform(WIDTH, HEIGHT);

        var input = Nd4j.create(nSamples, HEIGHT * WIDTH);
        var output = Nd4j.create(nSamples, nOutcomes);

        int n = 0;
        for (var folder : folders) {
            var imageFiles = folder.listFiles();
            var label = folder.getName();
            for (var imgFile : imageFiles) {
                var writableImg = nativeImageLoader.asWritable(imgFile);
                writableImg = resizer.transform(writableImg);
                var img = nativeImageLoader.asRowVector(writableImg.getFrame());
                scaler.transform(img);
                input.putRow(n, img);
                var labelNumber = labelToLabelNumber.get(label);
                output.put(n, labelNumber, 1.0);
                n++;
            }
        }
        var dataSet = new DataSet(input, output);
        var listDataSet = dataSet.asList();
        Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
        return new ListDataSetIterator<>(listDataSet, 10);
    }

    public void trainModel(DataSetIterator trainingDataSetIterator) {
        System.out.println("Train Model...");
        model.fit(trainingDataSetIterator);
    }

    public void evaluatingModel(DataSetIterator testingDataSetIterator) {
        System.out.println("Evaluating Model...");
        var eval = model.evaluate(testingDataSetIterator);
        System.out.println(eval.stats());
    }
}