package Homework1;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

public class k_means_clustering {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("KMeansClustering").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf); // connection to spark

        JavaRDD<String> lines = sc.textFile("uber_small.csv"); // each line becomes a single element in RDD

        JavaRDD<Tuple2<Vector, String>> groupRDD = lines.map(
            new Function<String, Tuple2<Vector, String>>() {
                //(vector of lat and lon, label)
                    public Tuple2<Vector, String> call(String line) {
                        String[] parts = line.split(",");
                        double lat = Double.parseDouble(parts[0]);
                        double lon = Double.parseDouble(parts[1]);
                        String group = parts[2];
                        Vector point = Vectors.dense(lat, lon);
                        return new Tuple2<>(point, group);
                    }
                }
        );
        for (Tuple2<Vector, String> tuple : groupRDD.take(5)) {
            System.out.println("Point: " + tuple._1 + " Group: " + tuple._2);
        }


    }
}
