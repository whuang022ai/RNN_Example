package com.mycompany.rnn_example;

import java.util.ArrayList;

/**
 * A Basic Sample RNN from scratch
 *
 * @author whuang022
 */
public class RNN_Basic {

    public ArrayList<Trainset> traindatas = new ArrayList<Trainset>();
    public double[] h1;
    public double[] h2;
    public double[] y;
    public double[] dy;
    public double[] dh1;
    public double[] dh2;
    public double[] sum_h1;
    public double[] sum_h2;
    public double[] sum_y;
    public double[] dsum_h1;
    public double[] dsum_h2;
    public double[] dsum_y;
    public int timesteps = 0;
    public double w11 = Math.random() * 2 - 1;//x1
    public double w12 = Math.random() * 2 - 1;//x2
    public double w21 = Math.random() * 2 - 1;//x1
    public double w22 = Math.random() * 2 - 1;//x2
    public double b11 = Math.random() * 2 - 1;
    public double b21 = Math.random() * 2 - 1;
    public double w13 = Math.random() * 2 - 1;//ht-1 to t
    public double w13_2 = Math.random() * 2 - 1;//ht-1 to t
    public double w23 = Math.random() * 2 - 1;
    public double w23_2 = Math.random() * 2 - 1;//ht-1 to t
    public double w14 = Math.random() * 2 - 1;
    public double w24 = Math.random() * 2 - 1;
    public double b2 = Math.random() * 2 - 1;

    public RNN_Basic(int timesteps) {

        this.timesteps = timesteps;
        h1 = new double[this.timesteps];
        h2 = new double[this.timesteps];
        y = new double[this.timesteps];
        dy = new double[this.timesteps];
        dh1 = new double[this.timesteps];
        dh2 = new double[this.timesteps];
        sum_h1 = new double[this.timesteps];
        sum_h2 = new double[this.timesteps];
        sum_y = new double[this.timesteps];
        dsum_y = new double[this.timesteps];
        dsum_h1 = new double[this.timesteps];
        dsum_h2 = new double[this.timesteps];

    }

    public double leakrelu(double x) {
        if (x >= 0) {
            return x;
        } else {
            return x * 0.001;
        }
    }

    public double leakrelu_div(double y) {
        if (y >= 0) {
            return 1;
        } else {
            return 0.001;
        }
    }

    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double sigmoid_div(double y) {
        return y * (1 - y);
    }

    public double tanh_div(double y) {
        return (1 - y * y);
    }

    public void forward(Trainset data, Boolean binout, Boolean out) {
        if (out) {
            System.out.println("[Outputs]");
        }
        for (int i = 0; i < timesteps; i++) {//for each time step
            double sum_h1t = 0;
            double sum_h2t = 0;
            double sum_yt = 0;
            if (i - 1 > 0) {
                sum_h1t = w11 * data.x1.get(i) + w12 * data.x2.get(i) + w13 * h1[i - 1] + w13_2 * h2[i - 1] + b11;
                sum_h2t = w21 * data.x1.get(i) + w22 * data.x2.get(i) + w23 * h2[i - 1] + w23_2 * h1[i - 1] + b21;
            } else {
                sum_h1t = w11 * data.x1.get(i) + w12 * data.x2.get(i) + w13 * 0 + w13_2 * 0 + b11;//  ht=0 = 0
                sum_h2t = w21 * data.x1.get(i) + w22 * data.x2.get(i) + w23 * 0 + w23_2 * 0 + b21;
            }
            h1[i] = Math.tanh(sum_h1t);
            h2[i] = Math.tanh(sum_h2t);
            sum_h1[i] = sum_h1t;
            sum_h2[i] = sum_h2t;
            sum_yt = w14 * sum_h1t + w24 * sum_h2t + b2;
            sum_y[i] = sum_yt;
            y[i] = sigmoid(sum_yt);
            // System.out.print("Y" + i + "=" +y[i] + " ");
            if (binout && out) {
                if (y[i] > 0.5) {
                    System.out.print("Y" + i + "=" + 1.0 + " ");
                } else {
                    System.out.print("Y" + i + "=" + 0.0 + " ");
                }
            } else {
                //System.out.print("Y" + i + "=" + y[i] + " ");
            }

        }
        //System.out.println("\n");
        if (out) {
            System.out.println("\n");
        }
    }

    public double backward(Trainset data) {
        //System.out.println("[Errors]");
        double sumESquare = 0;
        for (int i = timesteps - 1; i >= 0; i--) {//for each time step
            dy[i] = -(data.y1_desire.get(i) - y[i]); //mse error grident
            sumESquare += dy[i] * dy[i];
            //System.out.print("E" + i + "=" + dy[i] * dy[i] + " ");
            dsum_y[i] = dy[i] * sigmoid_div(y[i]);
            if (i == timesteps - 1) {
                dh1[i] = dsum_y[i] * w14 + 0 * w13;
                dh2[i] = dsum_y[i] * w24 + 0 * w23;
            } else {
                dh1[i] = dsum_y[i] * w14 + dh1[i + 1] * w13 + dh2[i + 1] * w13_2;
                dh2[i] = dsum_y[i] * w24 + dh2[i + 1] * w23 + dh1[i + 1] * w23_2;
            }
            dsum_h1[i] = dh1[i] * tanh_div(h1[i]);
            dsum_h2[i] = dh2[i] * tanh_div(h2[i]);
        }
        return sumESquare;
        //System.out.println("Error Square Sum =" + sumESquare);
        //System.out.println("\n");
    }

    public void update(Trainset data, double lr) {
        double dw14 = 0;
        double dw24 = 0;
        double db2 = 0;
        double dw13 = 0;
        double dw13_2 = 0;
        double dw23 = 0;
        double dw23_2 = 0;
        double dw11 = 0;
        double dw12 = 0;
        double dw21 = 0;
        double dw22 = 0;
        double db11 = 0;
        double db21 = 0;
        for (int i = 0; i < timesteps; i++) {
            dw14 += dsum_y[i] * h1[i];
            dw24 += dsum_y[i] * h2[i];
            db2 += dsum_y[i] * 1;
            if (i - 1 > 0) {
                dw13 += dsum_h1[i] * h1[i - 1];
                dw13_2 += dsum_h2[i] * h2[i - 1];
                dw23 += dsum_h2[i] * h2[i - 1];
                dw23_2 += dsum_h1[i] * h1[i - 1];
            } else {
                //  dw13 += dsum_h1[i] * 0;
                //  dw23 += dsum_h2[i] * 0;
            }
            dw11 += dsum_h1[i] * data.x1.get(i);
            dw12 += dsum_h1[i] * data.x2.get(i);

            dw21 += dsum_h2[i] * data.x1.get(i);
            dw22 += dsum_h2[i] * data.x2.get(i);

            db11 += dsum_h1[i] * 1;
            db21 += dsum_h2[i] * 1;
        }
        w14 -= dw14 * lr;
        w13 -= dw13 * lr;
        w13_2 -= dw13_2 * lr;

        w24 -= dw24 * lr;
        w23 -= dw23 * lr;
        w23_2 -= dw23_2 * lr;

        w12 -= dw12 * lr;
        w11 -= dw11 * lr;

        w22 -= dw22 * lr;
        w21 -= dw21 * lr;

        b11 -= db11 * lr;
        b21 -= db21 * lr;

        b2 -= db2 * lr;
    }

}
