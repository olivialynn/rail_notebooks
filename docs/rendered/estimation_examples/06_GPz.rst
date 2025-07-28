GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.3684247e-01	 3.1843928e-01	-3.2714755e-01	 3.2907435e-01	[-3.4703605e-01]	 4.6054411e-01


.. parsed-literal::

       2	-2.6509635e-01	 3.0743968e-01	-2.4119140e-01	 3.2008020e-01	[-2.7663803e-01]	 2.2904849e-01


.. parsed-literal::

       3	-2.2105968e-01	 2.8748893e-01	-1.8000621e-01	 2.9919930e-01	[-2.2597457e-01]	 2.8789186e-01


.. parsed-literal::

       4	-1.9201167e-01	 2.6356798e-01	-1.5213880e-01	 2.7014003e-01	[-1.9020361e-01]	 2.0847130e-01
       5	-1.0117991e-01	 2.5616434e-01	-6.6087195e-02	 2.6623564e-01	[-1.0130163e-01]	 1.9672441e-01


.. parsed-literal::

       6	-6.5823573e-02	 2.4995607e-01	-3.3990994e-02	 2.5696347e-01	[-5.9121668e-02]	 2.1459985e-01


.. parsed-literal::

       7	-4.7588223e-02	 2.4747007e-01	-2.3345165e-02	 2.5534424e-01	[-5.0681094e-02]	 2.1629477e-01
       8	-3.4075911e-02	 2.4517489e-01	-1.3788317e-02	 2.5383268e-01	[-4.2991328e-02]	 1.8059278e-01


.. parsed-literal::

       9	-1.8912547e-02	 2.4233531e-01	-1.7646055e-03	 2.5214261e-01	[-3.5631980e-02]	 2.1635747e-01
      10	-1.2103388e-02	 2.4133698e-01	 2.9768241e-03	 2.5015742e-01	[-2.4447909e-02]	 1.7640758e-01


.. parsed-literal::

      11	-5.7975263e-03	 2.4003357e-01	 8.6475953e-03	 2.4894734e-01	[-2.1912448e-02]	 1.8915200e-01
      12	-3.3589749e-03	 2.3954527e-01	 1.0919548e-02	 2.4890511e-01	[-2.0818769e-02]	 1.8679619e-01


.. parsed-literal::

      13	 3.5843950e-04	 2.3882453e-01	 1.4491366e-02	 2.4867584e-01	[-1.9044694e-02]	 2.0094132e-01


.. parsed-literal::

      14	 6.4227818e-03	 2.3759942e-01	 2.0956807e-02	 2.4756906e-01	[-1.2600224e-02]	 2.0824909e-01


.. parsed-literal::

      15	 1.1384140e-01	 2.2078440e-01	 1.3471068e-01	 2.3510560e-01	[ 1.1029389e-01]	 3.2070374e-01
      16	 2.3136365e-01	 2.1779945e-01	 2.5888345e-01	 2.2739562e-01	[ 2.3223459e-01]	 1.9906473e-01


.. parsed-literal::

      17	 2.8090994e-01	 2.1281487e-01	 3.1212770e-01	 2.1915118e-01	[ 2.9156787e-01]	 1.9467378e-01
      18	 3.2438818e-01	 2.0875842e-01	 3.5529010e-01	 2.1602261e-01	[ 3.2958086e-01]	 1.9744420e-01


.. parsed-literal::

      19	 3.7403455e-01	 2.0552143e-01	 4.0535710e-01	 2.1185157e-01	[ 3.8268053e-01]	 1.8250799e-01
      20	 4.6636834e-01	 1.9963793e-01	 4.9877328e-01	 2.0863964e-01	[ 4.6705982e-01]	 1.9332385e-01


.. parsed-literal::

      21	 5.3698906e-01	 2.0747340e-01	 5.7817761e-01	 2.1956913e-01	[ 5.1252185e-01]	 1.9588089e-01


.. parsed-literal::

      22	 5.9972291e-01	 2.0794542e-01	 6.3960044e-01	 2.2091351e-01	[ 5.8764235e-01]	 2.0596266e-01


.. parsed-literal::

      23	 6.3184747e-01	 1.9962613e-01	 6.6935512e-01	 2.1381315e-01	[ 6.1619551e-01]	 2.1158147e-01


.. parsed-literal::

      24	 6.6313904e-01	 1.9440933e-01	 6.9836872e-01	 2.0773796e-01	[ 6.4317310e-01]	 3.3140254e-01
      25	 6.8421416e-01	 1.9459556e-01	 7.1957923e-01	 2.0871904e-01	[ 6.5719739e-01]	 1.8489122e-01


.. parsed-literal::

      26	 7.2564294e-01	 1.9338416e-01	 7.6231229e-01	 2.0830282e-01	[ 6.9827026e-01]	 2.0518422e-01
      27	 7.5908647e-01	 1.9489166e-01	 7.9613735e-01	 2.0934906e-01	[ 7.3629559e-01]	 1.7525101e-01


.. parsed-literal::

      28	 7.8351089e-01	 1.9490062e-01	 8.2020329e-01	 2.0882557e-01	[ 7.5836730e-01]	 2.0052719e-01


.. parsed-literal::

      29	 8.0035796e-01	 1.9506491e-01	 8.3785620e-01	 2.0929574e-01	[ 7.7277684e-01]	 2.1568322e-01


.. parsed-literal::

      30	 8.2173758e-01	 1.9608030e-01	 8.6056485e-01	 2.1085053e-01	[ 7.8883501e-01]	 2.0347261e-01


.. parsed-literal::

      31	 8.5231802e-01	 1.9671939e-01	 8.9287990e-01	 2.1221625e-01	[ 8.1474234e-01]	 2.0893049e-01


.. parsed-literal::

      32	 8.6523774e-01	 1.9723797e-01	 9.0556256e-01	 2.1213665e-01	[ 8.2101111e-01]	 3.2070994e-01


.. parsed-literal::

      33	 8.8320746e-01	 1.9892190e-01	 9.2549659e-01	 2.1543885e-01	[ 8.2351614e-01]	 2.0851064e-01


.. parsed-literal::

      34	 9.0583523e-01	 1.9679921e-01	 9.4746575e-01	 2.1338545e-01	[ 8.5054889e-01]	 2.0899820e-01


.. parsed-literal::

      35	 9.2057287e-01	 1.9628611e-01	 9.6214934e-01	 2.1274567e-01	[ 8.7027681e-01]	 2.1296072e-01


.. parsed-literal::

      36	 9.3353496e-01	 1.9579386e-01	 9.7503050e-01	 2.1294793e-01	[ 8.8410361e-01]	 2.1892929e-01


.. parsed-literal::

      37	 9.4261546e-01	 1.9626595e-01	 9.8474712e-01	 2.1475030e-01	[ 8.9418828e-01]	 2.0970488e-01
      38	 9.5355583e-01	 1.9631855e-01	 9.9583294e-01	 2.1448096e-01	[ 8.9985733e-01]	 1.9181061e-01


.. parsed-literal::

      39	 9.6251519e-01	 1.9594838e-01	 1.0052227e+00	 2.1342793e-01	[ 9.0687845e-01]	 2.1663761e-01


.. parsed-literal::

      40	 9.7603741e-01	 1.9549311e-01	 1.0195170e+00	 2.1205486e-01	[ 9.2028055e-01]	 2.0765567e-01


.. parsed-literal::

      41	 9.8800261e-01	 1.9379238e-01	 1.0327063e+00	 2.0955821e-01	[ 9.3293815e-01]	 2.1594381e-01


.. parsed-literal::

      42	 9.9771216e-01	 1.9250890e-01	 1.0424695e+00	 2.0817286e-01	[ 9.4268233e-01]	 2.1827197e-01
      43	 1.0046092e+00	 1.9082084e-01	 1.0493113e+00	 2.0643356e-01	[ 9.4680068e-01]	 1.9134164e-01


.. parsed-literal::

      44	 1.0175763e+00	 1.8707154e-01	 1.0623119e+00	 2.0258346e-01	[ 9.5658289e-01]	 2.0802712e-01
      45	 1.0306713e+00	 1.8483083e-01	 1.0759545e+00	 2.0185912e-01	  9.5531106e-01 	 1.7394185e-01


.. parsed-literal::

      46	 1.0410177e+00	 1.8506121e-01	 1.0861198e+00	 2.0125231e-01	[ 9.7626734e-01]	 2.0650291e-01


.. parsed-literal::

      47	 1.0468556e+00	 1.8464290e-01	 1.0917772e+00	 2.0089740e-01	[ 9.8327026e-01]	 2.1280050e-01
      48	 1.0554345e+00	 1.8352784e-01	 1.1005684e+00	 2.0037667e-01	[ 9.9216306e-01]	 1.8193769e-01


.. parsed-literal::

      49	 1.0642436e+00	 1.8301717e-01	 1.1097490e+00	 2.0072699e-01	[ 9.9613979e-01]	 2.0346284e-01


.. parsed-literal::

      50	 1.0753996e+00	 1.8180233e-01	 1.1211863e+00	 1.9996829e-01	[ 1.0088698e+00]	 2.1783304e-01


.. parsed-literal::

      51	 1.0879250e+00	 1.7927531e-01	 1.1339965e+00	 1.9788041e-01	[ 1.0154160e+00]	 2.1465206e-01


.. parsed-literal::

      52	 1.0920263e+00	 1.7722198e-01	 1.1391825e+00	 1.9655726e-01	  1.0000379e+00 	 2.1256685e-01
      53	 1.1035829e+00	 1.7556234e-01	 1.1502047e+00	 1.9488338e-01	  1.0123584e+00 	 2.0216680e-01


.. parsed-literal::

      54	 1.1075920e+00	 1.7472347e-01	 1.1541502e+00	 1.9408314e-01	  1.0150225e+00 	 1.8604708e-01


.. parsed-literal::

      55	 1.1157037e+00	 1.7376444e-01	 1.1626536e+00	 1.9362623e-01	  1.0119918e+00 	 2.0825362e-01


.. parsed-literal::

      56	 1.1245331e+00	 1.7143942e-01	 1.1716366e+00	 1.9196697e-01	[ 1.0190347e+00]	 2.1309781e-01


.. parsed-literal::

      57	 1.1327739e+00	 1.7092494e-01	 1.1801608e+00	 1.9186698e-01	[ 1.0213601e+00]	 2.1439695e-01
      58	 1.1417161e+00	 1.7019076e-01	 1.1895154e+00	 1.9140673e-01	[ 1.0251594e+00]	 1.9918704e-01


.. parsed-literal::

      59	 1.1502699e+00	 1.6909167e-01	 1.1983069e+00	 1.9009770e-01	[ 1.0272529e+00]	 2.1496654e-01


.. parsed-literal::

      60	 1.1589335e+00	 1.6784362e-01	 1.2070714e+00	 1.8815040e-01	[ 1.0373010e+00]	 2.1639276e-01


.. parsed-literal::

      61	 1.1688248e+00	 1.6584953e-01	 1.2173903e+00	 1.8550307e-01	[ 1.0400548e+00]	 2.1925783e-01


.. parsed-literal::

      62	 1.1758675e+00	 1.6505576e-01	 1.2247900e+00	 1.8369524e-01	  1.0368251e+00 	 2.0485568e-01
      63	 1.1837643e+00	 1.6427471e-01	 1.2325897e+00	 1.8316241e-01	[ 1.0429465e+00]	 2.1531105e-01


.. parsed-literal::

      64	 1.1897954e+00	 1.6374566e-01	 1.2387366e+00	 1.8278634e-01	[ 1.0450932e+00]	 2.1726680e-01


.. parsed-literal::

      65	 1.1962501e+00	 1.6315643e-01	 1.2454097e+00	 1.8186222e-01	[ 1.0475217e+00]	 2.0925045e-01


.. parsed-literal::

      66	 1.2035282e+00	 1.6242347e-01	 1.2535250e+00	 1.8038347e-01	  1.0416586e+00 	 2.1697640e-01
      67	 1.2097636e+00	 1.6247730e-01	 1.2597846e+00	 1.8008034e-01	  1.0473665e+00 	 1.8127251e-01


.. parsed-literal::

      68	 1.2140557e+00	 1.6235322e-01	 1.2639551e+00	 1.7991772e-01	  1.0464500e+00 	 1.8932819e-01
      69	 1.2168442e+00	 1.6216403e-01	 1.2666861e+00	 1.7991206e-01	[ 1.0478305e+00]	 1.9004583e-01


.. parsed-literal::

      70	 1.2211821e+00	 1.6208947e-01	 1.2711712e+00	 1.7999576e-01	  1.0405937e+00 	 1.8743896e-01


.. parsed-literal::

      71	 1.2250195e+00	 1.6288159e-01	 1.2752264e+00	 1.8120979e-01	  1.0393159e+00 	 2.0549393e-01
      72	 1.2296809e+00	 1.6274344e-01	 1.2799362e+00	 1.8123479e-01	  1.0344913e+00 	 2.0043921e-01


.. parsed-literal::

      73	 1.2346429e+00	 1.6263064e-01	 1.2850231e+00	 1.8142322e-01	  1.0298981e+00 	 2.1658373e-01


.. parsed-literal::

      74	 1.2385888e+00	 1.6243438e-01	 1.2890327e+00	 1.8160926e-01	  1.0317835e+00 	 2.1712470e-01
      75	 1.2408004e+00	 1.6059698e-01	 1.2918267e+00	 1.8062159e-01	  1.0196050e+00 	 2.0138049e-01


.. parsed-literal::

      76	 1.2472662e+00	 1.6064662e-01	 1.2978798e+00	 1.8078230e-01	  1.0371738e+00 	 1.8373203e-01


.. parsed-literal::

      77	 1.2495399e+00	 1.6036990e-01	 1.3001032e+00	 1.8052271e-01	  1.0380452e+00 	 2.0806384e-01


.. parsed-literal::

      78	 1.2536535e+00	 1.5931274e-01	 1.3043150e+00	 1.7950194e-01	  1.0350448e+00 	 2.1568918e-01


.. parsed-literal::

      79	 1.2582885e+00	 1.5825800e-01	 1.3092118e+00	 1.7821168e-01	  1.0260027e+00 	 2.2079420e-01


.. parsed-literal::

      80	 1.2630684e+00	 1.5628949e-01	 1.3143788e+00	 1.7623455e-01	  1.0115095e+00 	 2.1403718e-01


.. parsed-literal::

      81	 1.2666023e+00	 1.5564855e-01	 1.3180571e+00	 1.7560121e-01	  9.9813091e-01 	 2.0404124e-01


.. parsed-literal::

      82	 1.2698879e+00	 1.5545250e-01	 1.3212368e+00	 1.7540883e-01	  1.0005891e+00 	 2.1691823e-01
      83	 1.2737030e+00	 1.5494656e-01	 1.3251396e+00	 1.7504529e-01	  9.9463272e-01 	 1.8687129e-01


.. parsed-literal::

      84	 1.2772707e+00	 1.5378690e-01	 1.3289948e+00	 1.7364352e-01	  9.7310271e-01 	 2.1137953e-01
      85	 1.2810259e+00	 1.5350941e-01	 1.3327379e+00	 1.7358632e-01	  9.6913995e-01 	 1.9697785e-01


.. parsed-literal::

      86	 1.2841141e+00	 1.5307706e-01	 1.3358693e+00	 1.7329557e-01	  9.6513452e-01 	 2.0994377e-01


.. parsed-literal::

      87	 1.2874359e+00	 1.5284105e-01	 1.3391926e+00	 1.7316156e-01	  9.6828023e-01 	 2.1968627e-01


.. parsed-literal::

      88	 1.2903253e+00	 1.5195282e-01	 1.3421785e+00	 1.7234647e-01	  9.6908772e-01 	 2.9473329e-01
      89	 1.2937308e+00	 1.5186686e-01	 1.3455605e+00	 1.7228011e-01	  9.7666266e-01 	 1.8251348e-01


.. parsed-literal::

      90	 1.2960915e+00	 1.5150484e-01	 1.3480073e+00	 1.7189269e-01	  9.7990296e-01 	 2.0763946e-01


.. parsed-literal::

      91	 1.2991235e+00	 1.5103039e-01	 1.3512238e+00	 1.7140130e-01	  9.7773299e-01 	 2.1531463e-01
      92	 1.3020581e+00	 1.5045658e-01	 1.3541962e+00	 1.7088884e-01	  9.8754987e-01 	 2.0242906e-01


.. parsed-literal::

      93	 1.3042734e+00	 1.5018237e-01	 1.3563874e+00	 1.7073170e-01	  9.8938672e-01 	 2.1065998e-01


.. parsed-literal::

      94	 1.3084176e+00	 1.4986808e-01	 1.3605683e+00	 1.7097237e-01	  9.9745065e-01 	 2.1888614e-01


.. parsed-literal::

      95	 1.3101150e+00	 1.4946504e-01	 1.3622861e+00	 1.7087537e-01	  9.9626149e-01 	 3.2231069e-01


.. parsed-literal::

      96	 1.3121778e+00	 1.4931129e-01	 1.3643822e+00	 1.7098571e-01	  1.0054110e+00 	 2.2097087e-01
      97	 1.3152893e+00	 1.4895468e-01	 1.3675490e+00	 1.7105513e-01	  1.0178833e+00 	 1.8653440e-01


.. parsed-literal::

      98	 1.3176530e+00	 1.4838040e-01	 1.3699495e+00	 1.7065928e-01	  1.0255230e+00 	 2.0361686e-01


.. parsed-literal::

      99	 1.3203634e+00	 1.4815147e-01	 1.3728414e+00	 1.7085606e-01	  1.0153034e+00 	 2.2104311e-01


.. parsed-literal::

     100	 1.3237849e+00	 1.4697269e-01	 1.3761977e+00	 1.6968003e-01	  1.0188073e+00 	 2.1257949e-01
     101	 1.3257383e+00	 1.4666032e-01	 1.3781040e+00	 1.6930038e-01	  1.0142054e+00 	 2.0478582e-01


.. parsed-literal::

     102	 1.3288508e+00	 1.4607055e-01	 1.3812542e+00	 1.6889411e-01	  9.9802723e-01 	 2.1263957e-01


.. parsed-literal::

     103	 1.3311878e+00	 1.4538759e-01	 1.3838878e+00	 1.6855573e-01	  9.9706179e-01 	 2.1342301e-01


.. parsed-literal::

     104	 1.3346969e+00	 1.4509343e-01	 1.3873298e+00	 1.6836806e-01	  9.9009611e-01 	 2.0809388e-01


.. parsed-literal::

     105	 1.3372606e+00	 1.4481038e-01	 1.3899640e+00	 1.6822888e-01	  9.9225028e-01 	 2.0794320e-01


.. parsed-literal::

     106	 1.3397990e+00	 1.4449198e-01	 1.3925908e+00	 1.6773869e-01	  9.9943538e-01 	 2.0363951e-01
     107	 1.3440672e+00	 1.4371007e-01	 1.3970614e+00	 1.6644080e-01	  1.0055902e+00 	 1.8472004e-01


.. parsed-literal::

     108	 1.3455075e+00	 1.4355545e-01	 1.3987736e+00	 1.6518775e-01	  1.0355608e+00 	 2.0977998e-01


.. parsed-literal::

     109	 1.3490807e+00	 1.4347040e-01	 1.4021329e+00	 1.6527788e-01	  1.0254154e+00 	 2.0369506e-01


.. parsed-literal::

     110	 1.3505684e+00	 1.4338671e-01	 1.4035818e+00	 1.6526207e-01	  1.0193796e+00 	 2.0407009e-01


.. parsed-literal::

     111	 1.3531953e+00	 1.4352770e-01	 1.4062071e+00	 1.6536047e-01	  1.0137980e+00 	 2.0809460e-01


.. parsed-literal::

     112	 1.3540834e+00	 1.4265324e-01	 1.4073078e+00	 1.6436918e-01	  1.0081236e+00 	 2.1018696e-01


.. parsed-literal::

     113	 1.3573441e+00	 1.4315555e-01	 1.4104519e+00	 1.6481035e-01	  1.0137503e+00 	 2.0652938e-01


.. parsed-literal::

     114	 1.3586085e+00	 1.4316468e-01	 1.4117473e+00	 1.6471530e-01	  1.0170857e+00 	 2.1527076e-01


.. parsed-literal::

     115	 1.3607146e+00	 1.4299372e-01	 1.4139410e+00	 1.6446191e-01	  1.0169056e+00 	 2.0432496e-01


.. parsed-literal::

     116	 1.3624188e+00	 1.4277205e-01	 1.4157979e+00	 1.6420282e-01	  1.0170387e+00 	 2.1734428e-01


.. parsed-literal::

     117	 1.3646719e+00	 1.4250015e-01	 1.4180478e+00	 1.6407723e-01	  1.0098917e+00 	 2.1528292e-01


.. parsed-literal::

     118	 1.3665102e+00	 1.4217809e-01	 1.4198903e+00	 1.6387970e-01	  1.0038533e+00 	 2.0887852e-01


.. parsed-literal::

     119	 1.3681079e+00	 1.4193323e-01	 1.4214840e+00	 1.6373509e-01	  1.0023091e+00 	 2.1071959e-01
     120	 1.3708315e+00	 1.4150963e-01	 1.4242275e+00	 1.6353638e-01	  1.0051421e+00 	 1.9932938e-01


.. parsed-literal::

     121	 1.3723769e+00	 1.4146383e-01	 1.4257741e+00	 1.6366707e-01	  1.0015520e+00 	 3.2342029e-01


.. parsed-literal::

     122	 1.3738876e+00	 1.4140906e-01	 1.4272799e+00	 1.6369286e-01	  1.0047827e+00 	 2.0896435e-01


.. parsed-literal::

     123	 1.3752745e+00	 1.4137892e-01	 1.4286707e+00	 1.6371515e-01	  1.0046139e+00 	 2.0833039e-01
     124	 1.3769377e+00	 1.4133734e-01	 1.4303511e+00	 1.6374539e-01	  1.0006477e+00 	 1.9877958e-01


.. parsed-literal::

     125	 1.3786746e+00	 1.4116319e-01	 1.4322394e+00	 1.6379207e-01	  9.8360811e-01 	 2.0902610e-01
     126	 1.3813174e+00	 1.4128418e-01	 1.4348580e+00	 1.6402656e-01	  9.8295591e-01 	 1.9926953e-01


.. parsed-literal::

     127	 1.3825927e+00	 1.4114193e-01	 1.4361243e+00	 1.6392504e-01	  9.8425413e-01 	 2.0952964e-01


.. parsed-literal::

     128	 1.3843448e+00	 1.4105442e-01	 1.4379219e+00	 1.6406528e-01	  9.8278812e-01 	 2.0786023e-01


.. parsed-literal::

     129	 1.3854899e+00	 1.4080107e-01	 1.4391188e+00	 1.6394743e-01	  9.8381056e-01 	 2.8656077e-01


.. parsed-literal::

     130	 1.3870778e+00	 1.4065999e-01	 1.4407219e+00	 1.6402582e-01	  9.8049527e-01 	 2.1187091e-01


.. parsed-literal::

     131	 1.3888822e+00	 1.4065222e-01	 1.4425413e+00	 1.6424625e-01	  9.7636156e-01 	 2.0587397e-01
     132	 1.3903838e+00	 1.4036895e-01	 1.4440118e+00	 1.6406233e-01	  9.7095486e-01 	 1.8907785e-01


.. parsed-literal::

     133	 1.3923980e+00	 1.4027019e-01	 1.4459974e+00	 1.6402950e-01	  9.6864262e-01 	 2.0671201e-01
     134	 1.3944461e+00	 1.3995710e-01	 1.4480861e+00	 1.6377272e-01	  9.7486682e-01 	 2.0306301e-01


.. parsed-literal::

     135	 1.3961557e+00	 1.3961822e-01	 1.4497670e+00	 1.6335350e-01	  9.6316293e-01 	 2.1370602e-01


.. parsed-literal::

     136	 1.3969892e+00	 1.3961016e-01	 1.4506002e+00	 1.6335609e-01	  9.6703124e-01 	 2.1230102e-01
     137	 1.3984513e+00	 1.3938248e-01	 1.4521215e+00	 1.6321708e-01	  9.6418435e-01 	 1.8761873e-01


.. parsed-literal::

     138	 1.4003764e+00	 1.3901369e-01	 1.4541688e+00	 1.6298863e-01	  9.4856936e-01 	 2.0982289e-01


.. parsed-literal::

     139	 1.4015201e+00	 1.3841660e-01	 1.4555323e+00	 1.6287508e-01	  8.9942763e-01 	 2.1341586e-01
     140	 1.4039048e+00	 1.3827616e-01	 1.4579109e+00	 1.6264992e-01	  8.9656181e-01 	 2.0027137e-01


.. parsed-literal::

     141	 1.4050305e+00	 1.3819980e-01	 1.4590257e+00	 1.6259546e-01	  8.9073422e-01 	 2.0625496e-01


.. parsed-literal::

     142	 1.4065753e+00	 1.3807652e-01	 1.4605729e+00	 1.6262385e-01	  8.7709255e-01 	 2.0588303e-01


.. parsed-literal::

     143	 1.4080358e+00	 1.3784567e-01	 1.4620869e+00	 1.6255351e-01	  8.7274696e-01 	 2.1975160e-01


.. parsed-literal::

     144	 1.4099498e+00	 1.3783697e-01	 1.4639485e+00	 1.6276823e-01	  8.6509024e-01 	 2.1273279e-01


.. parsed-literal::

     145	 1.4113638e+00	 1.3769741e-01	 1.4653871e+00	 1.6278207e-01	  8.6131939e-01 	 2.0551515e-01


.. parsed-literal::

     146	 1.4127714e+00	 1.3751152e-01	 1.4668295e+00	 1.6269531e-01	  8.6495034e-01 	 2.1540070e-01


.. parsed-literal::

     147	 1.4135180e+00	 1.3702020e-01	 1.4677434e+00	 1.6221462e-01	  8.4227121e-01 	 2.1255183e-01


.. parsed-literal::

     148	 1.4149961e+00	 1.3705343e-01	 1.4691756e+00	 1.6227781e-01	  8.6243927e-01 	 2.0668364e-01
     149	 1.4158298e+00	 1.3703416e-01	 1.4700002e+00	 1.6227501e-01	  8.6331367e-01 	 1.8577337e-01


.. parsed-literal::

     150	 1.4174963e+00	 1.3688598e-01	 1.4717237e+00	 1.6226617e-01	  8.5388296e-01 	 2.1440959e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.17 s, total: 2min 8s
    Wall time: 32.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fe0dc267d00>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.73 s, sys: 52 ms, total: 1.78 s
    Wall time: 547 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

