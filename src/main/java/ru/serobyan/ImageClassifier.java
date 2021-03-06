package ru.serobyan;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.*;

@Slf4j
public class ImageClassifier {

    private static final int HEIGHT = 64;
    private static final int WIDTH = 64;
    private static final double PREDICT_THRESHOLD = 0.8;

    private final File[] trainingImageFolders;
    private final File[] testingImageFolders;
    private final int nIncomes = HEIGHT * WIDTH * 3;
    private final int nOutcomes;
    private final Map<String, Integer> labelToLabelNumber;
    private final Map<Integer, String> labelNumberToLabel;
    private final NativeImageLoader nativeImageLoader = new NativeImageLoader();
    private final ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
    private final ResizeImageTransform resizer = new ResizeImageTransform(WIDTH, HEIGHT);
    private MultiLayerNetwork model;

    private ImageClassifier(
        File[] trainingImageFolders,
        File[] testingImageFolders,
        int nOutcomes,
        Map<String, Integer> labelToLabelNumber,
        Map<Integer, String> labelNumberToLabel
    ) {
        this.trainingImageFolders = trainingImageFolders;
        this.testingImageFolders = testingImageFolders;
        this.nOutcomes = nOutcomes;
        this.labelToLabelNumber = labelToLabelNumber;
        this.labelNumberToLabel = labelNumberToLabel;
    }

    public static ImageClassifier of(String trainingImageFoldersPath, String testingImageFoldersPath) {
        var trainingImageFolder = new File(trainingImageFoldersPath);
        var testingImageFolder = new File(testingImageFoldersPath);
        var trainingImageFolders = trainingImageFolder.listFiles();
        var testingImageFolders = testingImageFolder.listFiles();
        var nOutcomes = trainingImageFolders.length;
        var labelToLabelNumber = new HashMap<String, Integer>();
        var labelNumberToLabel = new HashMap<Integer, String>();
        var labelNumber = 0;
        for (var folder : trainingImageFolders) {
            var label = folder.getName();
            labelToLabelNumber.put(label, labelNumber);
            labelNumberToLabel.put(labelNumber, label);
            labelNumber++;
        }
        return new ImageClassifier(
            trainingImageFolders,
            testingImageFolders,
            nOutcomes,
            labelToLabelNumber,
            labelNumberToLabel
        );
    }

    public void buildModel() {
        log.info("Build Model...");
        var conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Nesterovs(0.006, 0.9))
            .l2(1e-4).list()
            .layer(
                new DenseLayer.Builder()
                    .nIn(nIncomes)
                    .nOut(1000)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .build()
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
        log.info(model.summary());
    }

    private DataSetIterator getDataSetIterator(File[] folders) {
        var nSamples = Arrays.stream(folders)
            .mapToInt(subFolder -> subFolder.listFiles().length)
            .sum();

        var input = Nd4j.create(nSamples, nIncomes);
        var output = Nd4j.create(nSamples, nOutcomes);

        int n = 0;
        for (var folder : folders) {
            var imageFiles = folder.listFiles();
            var label = folder.getName();
            for (var imageFile : imageFiles) {
                log.debug(n + ": " + imageFile.getName());
                var imageINDArray = imageToINDArray(imageFile);
                if (imageINDArray.isEmpty()) {
                    n++;
                    continue;
                }
                input.putRow(n, imageINDArray.get());
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

    @SneakyThrows
    private Optional<INDArray> imageToINDArray(File image) {
        if (image.getName().endsWith(".gif")) {
            log.warn("ACHTUNG!!!");
            return Optional.empty();
        }
        var writableImg = nativeImageLoader.asWritable(image);
        writableImg = resizer.transform(writableImg);
        var img = nativeImageLoader.asRowVector(writableImg.getFrame());
        if (img.data().length() != nIncomes) {
            log.error("ACHTUNG!!!");
            return Optional.empty();
        }
        scaler.transform(img);
        return Optional.of(img);
    }

    @SneakyThrows
    public void trainModel() {
        var ds = getDataSetIterator(trainingImageFolders);
        log.info("Train Model...");
        model.fit(ds);
    }

    @SneakyThrows
    public void evaluatingModel() {
        var ds = getDataSetIterator(testingImageFolders);
        log.info("Evaluating Model...");
        var eval = model.evaluate(ds);
        log.info(eval.stats());
    }

    @SneakyThrows
    public Optional<String> predict(File image) {
        log.debug("Predict image: " + image.getName());
        var imageINDArrayOpt = imageToINDArray(image);
        if (imageINDArrayOpt.isEmpty()) {
            return Optional.empty();
        }
        var input = Nd4j.create(1, nIncomes);
        input.putRow(0, imageINDArrayOpt.get());
        var results = model.output(input);
        var arrayResults = results.data().asDouble();
        log.debug("Predict results: " + Arrays.toString(arrayResults));
        var maxResult = 0.0;
        var maxLabelNumber = -1;
        var position = 0;
        for (var result : arrayResults) {
            if (result > maxResult) {
                maxResult = result;
                maxLabelNumber = position;
            }
            position++;
        }
        if (maxResult < PREDICT_THRESHOLD) {
            return Optional.empty();
        } else {
            return Optional.of(labelNumberToLabel.get(maxLabelNumber));
        }
    }
}