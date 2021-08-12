package ru.serobyan;

import java.io.File;

public class Main {

    public static void main(String[] args) {
        try {
//            var classifier = ImageClassifier.of("C:\\image\\training", "C:\\image\\testing");
            var classifier = ImageClassifier.of("C:\\mnist_png\\training", "C:\\mnist_png\\testing");
            classifier.buildModel();

            var trainingDataSetIterator = classifier.getTrainingDataSetIterator();
            classifier.trainModel(trainingDataSetIterator);

            var testingDataSetIterator = classifier.getTestingDataSetIterator();
            classifier.evaluatingModel(testingDataSetIterator);
            
            var label = classifier.predict(new File("C:\\mnist_png\\testing\\1\\5.png"));
            label.ifPresent(System.out::println);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
