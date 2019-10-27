package com.mycompany.rnn_example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;

/**
 * A Basic Sample RNN from scratch
 *
 */
public class App {

    //String.format("%16s", Integer.toBinaryString(1)).replace(' ', '0')
    public static String getbinary4digit(int input) {
        return String.format("%4s", Integer.toBinaryString(input)).replace(' ', '0');
    }

    public static double bin2double(char bin) {
        if (bin == '0') {
            return 0.0;
        } else if (bin == '1') {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    public static Trainset gen_Add_Trainset(int input_x1, int input_x2) {
        Trainset data = new Trainset();
        int output_y_desire = input_x1 + input_x2;
        String bin_x1 = getbinary4digit(input_x1);
        String bin_x2 = getbinary4digit(input_x2);
        String bin_y = getbinary4digit(output_y_desire);
        ArrayList<Double> bin_x1_double = new ArrayList<Double>();
        ArrayList<Double> bin_x2_double = new ArrayList<Double>();
        ArrayList<Double> bin_y_double = new ArrayList<Double>();
        for (int i = 3; i >= 0; i--) {
            char digitx1 = bin_x1.charAt(i);
            char digitx2 = bin_x2.charAt(i);
            char digity = bin_y.charAt(i);
            bin_x1_double.add(bin2double(digitx1));
            bin_x2_double.add(bin2double(digitx2));
            bin_y_double.add(bin2double(digity));
        }
        data.x1 = bin_x1_double;
        data.x2 = bin_x2_double;
        data.y1_desire = bin_y_double;
        System.out.println(data.x1);
        System.out.println(data.x2);
        System.out.println(data.y1_desire);
        return data;
    }

    public static void main(String[] args) {
        //RNN_Add4_example();
        RNN_Ser_example();
    }

    public static void RNN_Ser_example() {
        RNN_Basic rnn = new RNN_Basic(4);
        Trainset set1 = new Trainset();
        //
        set1.x1.add(0.0);
        set1.x1.add(2.0);
        set1.x1.add(4.0);
        set1.x1.add(6.0);

        set1.x2.add(2.0);
        set1.x2.add(4.0);
        set1.x2.add(6.0);
        set1.x2.add(8.0);

        set1.y1_desire.add(0.0);
        set1.y1_desire.add(0.0);
        set1.y1_desire.add(0.0);
        set1.y1_desire.add(0.0);
        rnn.traindatas.add(set1);

        //
        Trainset set2 = new Trainset();
        //
        set2.x1.add(0.0);
        set2.x1.add(4.0);
        set2.x1.add(8.0);
        set2.x1.add(12.0);

        set2.x2.add(4.0);
        set2.x2.add(8.0);
        set2.x2.add(12.0);
        set2.x2.add(16.0);

        set2.y1_desire.add(1.0);
        set2.y1_desire.add(1.0);
        set2.y1_desire.add(1.0);
        set2.y1_desire.add(1.0);
        rnn.traindatas.add(set2);
          //

        Trainset set3 = new Trainset();
        //
        set3.x1.add(8.0);
        set3.x1.add(10.0);
        set3.x1.add(12.0);
        set3.x1.add(14.0);

        set3.x2.add(10.0);
        set3.x2.add(12.0);
        set3.x2.add(14.0);
        set3.x2.add(16.0);

        set3.y1_desire.add(0.0);
        set3.y1_desire.add(0.0);
        set3.y1_desire.add(0.0);
        set3.y1_desire.add(0.0);
        rnn.traindatas.add(set3);

        Trainset set4 = new Trainset();
        //
        set4.x1.add(16.0);
        set4.x1.add(20.0);
        set4.x1.add(24.0);
        set4.x1.add(28.0);

        set4.x2.add(20.0);
        set4.x2.add(24.0);
        set4.x2.add(28.0);
        set4.x2.add(32.0);

        set4.y1_desire.add(1.0);
        set4.y1_desire.add(1.0);
        set4.y1_desire.add(1.0);
        set4.y1_desire.add(1.0);
        rnn.traindatas.add(set4);

        for (int i = 0; i < 2000; i++) {
            Collections.shuffle(rnn.traindatas);
            double errors = 0;
            for (int j = 0; j < rnn.traindatas.size(); j++) {
                Trainset sample = rnn.traindatas.get(j);
                rnn.forward(sample, false, false);
                errors += rnn.backward(sample);
                //if(errors<50){break;}
                rnn.update(sample, 0.01); //5000 0.05

            }
            double size = rnn.traindatas.size();
            errors /= size;
            // if(i%10==0){
            System.out.println(errors);
            // }

        }
    }

    public static void RNN_Add4_example() {
        RNN_Basic rnn = new RNN_Basic(4);

        for (int x1 = 0; x1 <= 15; x1++) {
            for (int x2 = 15; x2 >= 0; x2--) {
                if (x1 + x2 <= 15) {
                    Trainset data = gen_Add_Trainset(x1, x2);
                    rnn.traindatas.add(data);
                }
            }
        }

        //
        for (int i = 0; i < 301; i++) {
            Collections.shuffle(rnn.traindatas);
            double errors = 0;
            for (int j = 0; j < rnn.traindatas.size(); j++) {
                Trainset sample = rnn.traindatas.get(j);
                rnn.forward(sample, false, false);
                errors += rnn.backward(sample);
                //if(errors<50){break;}
                rnn.update(sample, 0.05); //5000 0.05

            }
            double size = rnn.traindatas.size();
            errors /= size;
            // if(i%10==0){
            System.out.println(errors);
            // }

        }

        while (true) {
            Scanner scan = new Scanner(System.in);
            System.out.println("[Input X1]");
            int x1 = scan.nextInt();
            System.out.println("[Input X2]");
            int x2 = scan.nextInt();
            Trainset tmp_data = gen_Add_Trainset(x1, x2);
            rnn.forward(tmp_data, true, true);
        }

    }
}
