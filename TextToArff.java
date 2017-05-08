package com;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

/**
convert data.txt to data.arff
**/
public class TextToArff
{
	/**
	 * 
	 * @param source 源文件(.txt文件)
	 * @param destination 目标文件(.arff)
	 * @throws Exception
	 */
	public void txtToArff(String source,String destination) throws Exception
	{
		//读入txt格式数据集
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(source));
		String[] options = {"-F<','>"};//‘，’为分隔符
		loader.setOptions(options);
		Instances data = loader.getDataSet();
		//保存为Arff格式数据集
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File(destination));
		saver.writeBatch();
	}
	
	public static void main(String[] args) throws Exception
	{
		// TODO Auto-generated method stub
		TextToArff tta = new TextToArff();
		tta.txtToArff("data//西瓜数据集2.0.txt", "data//西瓜数据集2.0.arff");
	}

}
