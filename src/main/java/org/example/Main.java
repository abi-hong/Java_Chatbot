package org.example;

import kr.co.shineware.nlp.komoran.constant.DEFAULT_MODEL;
import kr.co.shineware.nlp.komoran.core.Komoran;
import kr.co.shineware.nlp.komoran.model.KomoranResult;

import com.github.jfasttext.JFastText;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Main {
    public static void main(String[] args)throws IOException {

        /*
        Komoran으로 형태소 분석
        * */

        // FULL 모델을 갖는 Komoran 객체를 선언
        Komoran komoran = new Komoran(DEFAULT_MODEL.FULL);

        // 텍스트 파일 받기
        File file = new File("src/chatbot_training_data.txt");
        BufferedReader reader = new BufferedReader(new FileReader(file));

        String strToAnalyze;
        KomoranResult analyzeResultList;
        while ((strToAnalyze = reader.readLine()) != null) {

            // 형태소분석할 문장
            System.out.println("형태소 분석할 문장 : " + strToAnalyze);

            // Komoran 객체의 analyze()메소드의 인자로 분석할 문장을 전달
            // 이 결과를 KomoranResult 객체로 저장
            analyzeResultList = komoran.analyze(strToAnalyze);

            // 형태소 분석 결과 중 명사류를 List<String> 형태로 반환
            System.out.println("==========print 'getNouns()'==========");
            System.out.println(analyzeResultList.getNouns());
            System.out.println();
        }


        // JFastText 자연어 처리
        JFastText jft = new JFastText();

        // Train supervised model
        jft.runCmd(new String[] {
                "supervised",
                "-input", "src/chatbot_training_data.txt",
                "-output", "src/main/resources/models/lid.176.bin"
        });


        // Load model from file
        jft.loadModel("src/main/resources/models/lid.176.bin");

        // Do label prediction
        String text = "배달 가능 한가요 ?";
        JFastText.ProbLabel probLabel = jft.predictProba(text);
        System.out.printf("\nThe label of '%s' is '%s' with probability %f\n",
                text, probLabel.label, Math.exp(probLabel.logProb));
        System.out.println("probLabel stirng: " + probLabel.toString());
    }
}