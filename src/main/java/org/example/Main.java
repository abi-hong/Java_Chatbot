package org.example;

import kr.co.shineware.nlp.komoran.constant.DEFAULT_MODEL;
import kr.co.shineware.nlp.komoran.core.Komoran;
import kr.co.shineware.nlp.komoran.model.KomoranResult;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

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
    }
}