# machine-learning
personal practice
------
个人练习，该项目内容包括：

1.周志华《机器学习》---西瓜数据集2.0

2.TextToArff.java ---将data.txt转换为data.arff/convert data.txt to data.arff

3.ID3---支持可视化的ID3算法，ID3算法还是官方提供的，我只是在此基础上参考http://davis.wpi.edu/~xmdv/weka/ 加了个可视化函数而已，请放心使用。

展示可视化的代码：

```java
//可视化ID3生成的树
		TreeVisualizer tv = new TreeVisualizer(null, classifier.graph(), new PlaceNode2());
		JFrame jf = new JFrame("weka Classifier Tree Visualizer:ID3");
		jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		jf.setSize(700, 700);
		jf.getContentPane().setLayout(new BorderLayout());
		jf.getContentPane().add(tv, BorderLayout.CENTER);
		jf.setVisible(true);
		tv.fitToScreen();
		System.out.println("end");
```
运行效果如图所示:

![](doc-img/id3可视化.jpg)

<br>

4.博客《scikit-learn之kmeans应用及问题》代码，博客地址：https://blog.csdn.net/u012328159/article/details/86554833

动态更新.................
