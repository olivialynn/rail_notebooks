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

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4474389e-01	 3.2114900e-01	-3.3504859e-01	 3.1831477e-01	[-3.2892699e-01]	 4.7052121e-01


.. parsed-literal::

       2	-2.7414949e-01	 3.1020930e-01	-2.4975168e-01	 3.0930372e-01	[-2.4411737e-01]	 2.3442864e-01


.. parsed-literal::

       3	-2.3192058e-01	 2.9034885e-01	-1.8988128e-01	 2.9026539e-01	[-1.8724447e-01]	 2.8077316e-01
       4	-1.9047872e-01	 2.6619825e-01	-1.4751701e-01	 2.6736390e-01	[-1.5296971e-01]	 1.8246341e-01


.. parsed-literal::

       5	-1.0825779e-01	 2.5721960e-01	-7.5504789e-02	 2.6123619e-01	[-8.2005392e-02]	 2.0526361e-01


.. parsed-literal::

       6	-7.0675804e-02	 2.5159074e-01	-4.0612540e-02	 2.5169702e-01	[-3.9055124e-02]	 2.1270323e-01
       7	-5.5341062e-02	 2.4930886e-01	-3.0476427e-02	 2.4980716e-01	[-2.8530828e-02]	 1.9272542e-01


.. parsed-literal::

       8	-3.9779188e-02	 2.4669127e-01	-1.9513624e-02	 2.4749410e-01	[-1.7546756e-02]	 2.2335553e-01
       9	-2.7890311e-02	 2.4455926e-01	-1.0400020e-02	 2.4495688e-01	[-7.6298471e-03]	 1.9168019e-01


.. parsed-literal::

      10	-1.9289878e-02	 2.4335126e-01	-4.2416414e-03	 2.4284451e-01	[ 1.7479221e-03]	 2.1532083e-01


.. parsed-literal::

      11	-1.2768341e-02	 2.4191069e-01	 1.6234872e-03	 2.4058145e-01	[ 9.7608004e-03]	 2.2310424e-01


.. parsed-literal::

      12	-1.0666090e-02	 2.4154479e-01	 3.4374730e-03	 2.4017080e-01	[ 1.2219379e-02]	 2.1681690e-01


.. parsed-literal::

      13	-5.5304770e-03	 2.4056834e-01	 8.4069550e-03	 2.3967543e-01	[ 1.6966545e-02]	 2.2117639e-01


.. parsed-literal::

      14	 1.8328520e-01	 2.2453031e-01	 2.0975159e-01	 2.2748417e-01	[ 2.2023715e-01]	 4.4242454e-01
      15	 1.9005920e-01	 2.1974768e-01	 2.1785051e-01	 2.1494726e-01	[ 2.3034949e-01]	 1.8614936e-01


.. parsed-literal::

      16	 3.1279811e-01	 2.1203001e-01	 3.4346182e-01	 2.0632482e-01	[ 3.7084410e-01]	 2.0477200e-01
      17	 3.7296808e-01	 2.0573649e-01	 4.0810701e-01	 1.9779731e-01	[ 4.5703852e-01]	 1.9463491e-01


.. parsed-literal::

      18	 4.4859262e-01	 2.0203703e-01	 4.8349760e-01	 1.9459991e-01	[ 5.1592568e-01]	 2.0873666e-01


.. parsed-literal::

      19	 5.3631412e-01	 1.9665032e-01	 5.7276495e-01	 1.8890639e-01	[ 5.9359462e-01]	 2.0953298e-01


.. parsed-literal::

      20	 6.2589620e-01	 1.9489432e-01	 6.6872436e-01	 1.8838800e-01	[ 6.8302500e-01]	 2.0792532e-01
      21	 6.7906720e-01	 1.9067787e-01	 7.1884509e-01	 1.8404933e-01	[ 7.3002475e-01]	 1.9981742e-01


.. parsed-literal::

      22	 7.1354479e-01	 1.8699411e-01	 7.5268401e-01	 1.8071054e-01	[ 7.5828799e-01]	 2.1493554e-01
      23	 7.5360273e-01	 1.8597311e-01	 7.9278795e-01	 1.7939817e-01	[ 7.7896437e-01]	 1.8746519e-01


.. parsed-literal::

      24	 8.0441158e-01	 1.8750667e-01	 8.4396078e-01	 1.7714309e-01	[ 8.3399972e-01]	 2.1292090e-01


.. parsed-literal::

      25	 8.4342571e-01	 1.8588990e-01	 8.8387747e-01	 1.7594714e-01	[ 8.7610574e-01]	 2.1208382e-01


.. parsed-literal::

      26	 8.7358220e-01	 1.8790051e-01	 9.1408483e-01	 1.7901005e-01	[ 8.9387939e-01]	 2.2207165e-01


.. parsed-literal::

      27	 8.9725542e-01	 1.8572571e-01	 9.3848359e-01	 1.7690445e-01	[ 9.1069396e-01]	 2.2125936e-01


.. parsed-literal::

      28	 9.1896787e-01	 1.8338681e-01	 9.6123017e-01	 1.7517751e-01	[ 9.2233592e-01]	 2.0851016e-01


.. parsed-literal::

      29	 9.4812033e-01	 1.7814350e-01	 9.9211376e-01	 1.7056272e-01	[ 9.3873239e-01]	 2.1491289e-01


.. parsed-literal::

      30	 9.6789249e-01	 1.7388283e-01	 1.0128947e+00	 1.6673096e-01	[ 9.6215272e-01]	 2.1701312e-01
      31	 9.9036047e-01	 1.6785445e-01	 1.0358141e+00	 1.6076519e-01	[ 9.8468938e-01]	 1.9527507e-01


.. parsed-literal::

      32	 1.0055446e+00	 1.6461107e-01	 1.0507802e+00	 1.5649233e-01	[ 1.0023921e+00]	 2.0832372e-01
      33	 1.0226335e+00	 1.6172293e-01	 1.0680455e+00	 1.5300914e-01	[ 1.0259840e+00]	 1.8018532e-01


.. parsed-literal::

      34	 1.0433244e+00	 1.5951892e-01	 1.0889517e+00	 1.5149220e-01	[ 1.0457019e+00]	 1.9573879e-01


.. parsed-literal::

      35	 1.0621822e+00	 1.5684289e-01	 1.1081640e+00	 1.5062362e-01	[ 1.0654789e+00]	 2.0493007e-01


.. parsed-literal::

      36	 1.0749575e+00	 1.5416055e-01	 1.1214058e+00	 1.4924426e-01	[ 1.0747271e+00]	 2.1226168e-01
      37	 1.0968754e+00	 1.5122916e-01	 1.1441817e+00	 1.4699028e-01	[ 1.0848004e+00]	 1.8274832e-01


.. parsed-literal::

      38	 1.1143225e+00	 1.5078319e-01	 1.1613862e+00	 1.4651084e-01	[ 1.0985756e+00]	 1.9214106e-01


.. parsed-literal::

      39	 1.1245390e+00	 1.5004062e-01	 1.1719308e+00	 1.4515602e-01	  1.0984569e+00 	 2.1178484e-01


.. parsed-literal::

      40	 1.1364346e+00	 1.4818704e-01	 1.1835996e+00	 1.4418066e-01	[ 1.1127460e+00]	 2.1578312e-01
      41	 1.1483310e+00	 1.4667470e-01	 1.1954266e+00	 1.4316502e-01	[ 1.1272348e+00]	 2.0735526e-01


.. parsed-literal::

      42	 1.1584394e+00	 1.4554122e-01	 1.2053792e+00	 1.4229397e-01	[ 1.1370993e+00]	 2.0505023e-01


.. parsed-literal::

      43	 1.1710419e+00	 1.4471787e-01	 1.2182005e+00	 1.4221544e-01	[ 1.1467329e+00]	 2.0387602e-01


.. parsed-literal::

      44	 1.1845041e+00	 1.4323620e-01	 1.2314857e+00	 1.4042426e-01	[ 1.1530622e+00]	 2.2354937e-01


.. parsed-literal::

      45	 1.1909155e+00	 1.4267004e-01	 1.2378813e+00	 1.3954045e-01	[ 1.1587237e+00]	 2.1082807e-01


.. parsed-literal::

      46	 1.2119694e+00	 1.4081585e-01	 1.2594839e+00	 1.3530864e-01	[ 1.1722484e+00]	 2.1907282e-01


.. parsed-literal::

      47	 1.2221050e+00	 1.4008051e-01	 1.2702334e+00	 1.3542786e-01	[ 1.1813620e+00]	 2.1769977e-01
      48	 1.2336986e+00	 1.3917377e-01	 1.2814640e+00	 1.3433366e-01	[ 1.1937455e+00]	 1.9008875e-01


.. parsed-literal::

      49	 1.2431897e+00	 1.3847860e-01	 1.2912325e+00	 1.3363843e-01	[ 1.2006290e+00]	 1.9487643e-01


.. parsed-literal::

      50	 1.2530270e+00	 1.3790825e-01	 1.3015485e+00	 1.3357339e-01	[ 1.2046515e+00]	 2.1506715e-01


.. parsed-literal::

      51	 1.2621347e+00	 1.3790890e-01	 1.3117868e+00	 1.3393603e-01	[ 1.2074253e+00]	 2.1259332e-01
      52	 1.2720590e+00	 1.3752394e-01	 1.3218030e+00	 1.3384141e-01	[ 1.2079689e+00]	 1.9032168e-01


.. parsed-literal::

      53	 1.2771125e+00	 1.3709264e-01	 1.3265090e+00	 1.3344775e-01	[ 1.2139852e+00]	 2.1166205e-01
      54	 1.2877282e+00	 1.3659857e-01	 1.3371898e+00	 1.3323037e-01	[ 1.2206660e+00]	 1.7763662e-01


.. parsed-literal::

      55	 1.2988703e+00	 1.3574292e-01	 1.3486564e+00	 1.3263445e-01	[ 1.2212218e+00]	 2.0412326e-01


.. parsed-literal::

      56	 1.3062707e+00	 1.3602784e-01	 1.3560517e+00	 1.3342702e-01	[ 1.2301847e+00]	 2.1049142e-01


.. parsed-literal::

      57	 1.3137441e+00	 1.3517403e-01	 1.3634746e+00	 1.3214556e-01	[ 1.2371236e+00]	 2.1837068e-01
      58	 1.3204842e+00	 1.3454637e-01	 1.3703387e+00	 1.3112950e-01	[ 1.2422342e+00]	 2.0412016e-01


.. parsed-literal::

      59	 1.3284133e+00	 1.3375588e-01	 1.3786346e+00	 1.2964658e-01	[ 1.2479535e+00]	 2.1015072e-01


.. parsed-literal::

      60	 1.3335224e+00	 1.3387049e-01	 1.3842587e+00	 1.2822912e-01	[ 1.2595416e+00]	 2.0642161e-01


.. parsed-literal::

      61	 1.3405252e+00	 1.3317075e-01	 1.3910005e+00	 1.2745374e-01	[ 1.2647053e+00]	 2.0863676e-01
      62	 1.3451264e+00	 1.3274678e-01	 1.3954997e+00	 1.2664941e-01	[ 1.2688350e+00]	 1.7845225e-01


.. parsed-literal::

      63	 1.3510224e+00	 1.3230089e-01	 1.4013522e+00	 1.2558630e-01	[ 1.2754304e+00]	 2.1007013e-01


.. parsed-literal::

      64	 1.3587811e+00	 1.3132999e-01	 1.4094368e+00	 1.2290234e-01	  1.2743700e+00 	 2.0627618e-01


.. parsed-literal::

      65	 1.3661930e+00	 1.3092282e-01	 1.4168213e+00	 1.2223321e-01	[ 1.2829923e+00]	 2.2111535e-01


.. parsed-literal::

      66	 1.3709902e+00	 1.3060482e-01	 1.4216534e+00	 1.2207587e-01	[ 1.2839342e+00]	 2.1278644e-01


.. parsed-literal::

      67	 1.3794873e+00	 1.2957594e-01	 1.4307121e+00	 1.2121107e-01	  1.2740005e+00 	 2.0461917e-01
      68	 1.3819941e+00	 1.2958297e-01	 1.4333504e+00	 1.2065998e-01	  1.2709416e+00 	 1.9601607e-01


.. parsed-literal::

      69	 1.3881687e+00	 1.2908992e-01	 1.4393029e+00	 1.2050073e-01	  1.2775135e+00 	 1.9925141e-01
      70	 1.3920641e+00	 1.2873479e-01	 1.4432373e+00	 1.2015551e-01	  1.2795028e+00 	 1.9433594e-01


.. parsed-literal::

      71	 1.3958886e+00	 1.2835559e-01	 1.4471316e+00	 1.2001887e-01	  1.2833027e+00 	 2.1861839e-01


.. parsed-literal::

      72	 1.4017632e+00	 1.2764041e-01	 1.4531870e+00	 1.1939852e-01	[ 1.2884633e+00]	 2.1535563e-01


.. parsed-literal::

      73	 1.4055417e+00	 1.2716473e-01	 1.4571489e+00	 1.1960088e-01	[ 1.2938625e+00]	 3.3001924e-01


.. parsed-literal::

      74	 1.4105274e+00	 1.2678735e-01	 1.4621862e+00	 1.1927733e-01	[ 1.2997151e+00]	 2.1520782e-01


.. parsed-literal::

      75	 1.4149796e+00	 1.2651755e-01	 1.4667317e+00	 1.1916837e-01	[ 1.3043995e+00]	 2.1279621e-01


.. parsed-literal::

      76	 1.4194142e+00	 1.2640439e-01	 1.4712066e+00	 1.1914758e-01	[ 1.3076828e+00]	 2.1065164e-01
      77	 1.4243344e+00	 1.2599227e-01	 1.4761183e+00	 1.1913508e-01	[ 1.3085632e+00]	 1.9774652e-01


.. parsed-literal::

      78	 1.4283319e+00	 1.2595422e-01	 1.4800510e+00	 1.1935778e-01	  1.3083770e+00 	 2.1155858e-01
      79	 1.4320111e+00	 1.2568065e-01	 1.4837439e+00	 1.1914807e-01	[ 1.3102225e+00]	 1.9018412e-01


.. parsed-literal::

      80	 1.4370718e+00	 1.2514978e-01	 1.4888880e+00	 1.1898728e-01	[ 1.3147955e+00]	 2.1106291e-01


.. parsed-literal::

      81	 1.4414420e+00	 1.2448711e-01	 1.4934859e+00	 1.1819220e-01	  1.3138144e+00 	 2.1130991e-01


.. parsed-literal::

      82	 1.4450868e+00	 1.2416629e-01	 1.4970563e+00	 1.1790522e-01	[ 1.3192236e+00]	 2.1560955e-01


.. parsed-literal::

      83	 1.4474307e+00	 1.2397079e-01	 1.4993690e+00	 1.1762312e-01	[ 1.3210343e+00]	 2.2057700e-01


.. parsed-literal::

      84	 1.4503501e+00	 1.2368726e-01	 1.5022903e+00	 1.1707501e-01	  1.3205346e+00 	 2.1211576e-01
      85	 1.4530734e+00	 1.2319269e-01	 1.5050610e+00	 1.1618453e-01	  1.3206787e+00 	 1.9186354e-01


.. parsed-literal::

      86	 1.4564525e+00	 1.2312706e-01	 1.5083687e+00	 1.1589102e-01	  1.3197571e+00 	 2.0136380e-01


.. parsed-literal::

      87	 1.4584656e+00	 1.2304483e-01	 1.5103609e+00	 1.1579477e-01	  1.3192137e+00 	 2.1294165e-01


.. parsed-literal::

      88	 1.4615548e+00	 1.2293022e-01	 1.5134998e+00	 1.1556520e-01	  1.3174296e+00 	 2.1682215e-01


.. parsed-literal::

      89	 1.4637973e+00	 1.2260007e-01	 1.5160331e+00	 1.1494348e-01	  1.3114355e+00 	 2.0909524e-01


.. parsed-literal::

      90	 1.4681988e+00	 1.2237616e-01	 1.5204411e+00	 1.1449533e-01	  1.3119929e+00 	 2.1281481e-01
      91	 1.4699730e+00	 1.2217825e-01	 1.5222463e+00	 1.1416319e-01	  1.3132616e+00 	 2.0130992e-01


.. parsed-literal::

      92	 1.4721734e+00	 1.2191126e-01	 1.5245313e+00	 1.1363163e-01	  1.3133650e+00 	 2.1270943e-01


.. parsed-literal::

      93	 1.4757749e+00	 1.2160652e-01	 1.5281943e+00	 1.1301443e-01	  1.3119600e+00 	 2.1424484e-01


.. parsed-literal::

      94	 1.4783195e+00	 1.2142434e-01	 1.5308186e+00	 1.1257972e-01	  1.3109413e+00 	 3.2425165e-01


.. parsed-literal::

      95	 1.4820176e+00	 1.2127650e-01	 1.5345344e+00	 1.1232928e-01	  1.3070665e+00 	 2.0496917e-01


.. parsed-literal::

      96	 1.4847176e+00	 1.2119651e-01	 1.5372766e+00	 1.1234235e-01	  1.2983605e+00 	 2.0127320e-01
      97	 1.4868949e+00	 1.2108186e-01	 1.5395141e+00	 1.1239629e-01	  1.2939676e+00 	 2.0171285e-01


.. parsed-literal::

      98	 1.4889560e+00	 1.2098477e-01	 1.5415549e+00	 1.1252938e-01	  1.2897926e+00 	 2.0468903e-01


.. parsed-literal::

      99	 1.4906958e+00	 1.2083891e-01	 1.5433219e+00	 1.1249607e-01	  1.2863760e+00 	 2.0264316e-01
     100	 1.4926299e+00	 1.2068467e-01	 1.5452691e+00	 1.1245118e-01	  1.2825938e+00 	 1.8056560e-01


.. parsed-literal::

     101	 1.4938900e+00	 1.2051130e-01	 1.5466720e+00	 1.1285768e-01	  1.2641095e+00 	 2.1013212e-01


.. parsed-literal::

     102	 1.4965740e+00	 1.2046134e-01	 1.5492117e+00	 1.1261570e-01	  1.2707732e+00 	 2.1171546e-01


.. parsed-literal::

     103	 1.4976951e+00	 1.2045929e-01	 1.5502842e+00	 1.1254795e-01	  1.2720096e+00 	 2.1637988e-01


.. parsed-literal::

     104	 1.4994042e+00	 1.2045813e-01	 1.5519685e+00	 1.1253271e-01	  1.2732064e+00 	 2.1057200e-01
     105	 1.5009823e+00	 1.2049479e-01	 1.5536162e+00	 1.1272219e-01	  1.2723591e+00 	 1.7448187e-01


.. parsed-literal::

     106	 1.5029443e+00	 1.2045755e-01	 1.5556141e+00	 1.1267872e-01	  1.2722065e+00 	 2.1604919e-01


.. parsed-literal::

     107	 1.5040496e+00	 1.2036941e-01	 1.5567605e+00	 1.1269460e-01	  1.2709001e+00 	 2.2268939e-01


.. parsed-literal::

     108	 1.5059574e+00	 1.2023007e-01	 1.5587673e+00	 1.1277335e-01	  1.2699802e+00 	 2.0959640e-01


.. parsed-literal::

     109	 1.5067685e+00	 1.1997830e-01	 1.5597807e+00	 1.1273832e-01	  1.2522260e+00 	 2.0794439e-01


.. parsed-literal::

     110	 1.5090853e+00	 1.1997479e-01	 1.5619768e+00	 1.1268961e-01	  1.2627498e+00 	 2.1135616e-01
     111	 1.5105222e+00	 1.1994399e-01	 1.5633650e+00	 1.1262139e-01	  1.2655060e+00 	 1.7902398e-01


.. parsed-literal::

     112	 1.5121014e+00	 1.1983722e-01	 1.5649122e+00	 1.1246852e-01	  1.2640889e+00 	 2.1924520e-01
     113	 1.5125836e+00	 1.1973622e-01	 1.5654963e+00	 1.1222417e-01	  1.2642882e+00 	 1.9199777e-01


.. parsed-literal::

     114	 1.5151124e+00	 1.1959437e-01	 1.5679246e+00	 1.1216575e-01	  1.2604991e+00 	 2.2154665e-01


.. parsed-literal::

     115	 1.5161698e+00	 1.1948714e-01	 1.5690066e+00	 1.1209653e-01	  1.2573796e+00 	 2.1381783e-01
     116	 1.5174477e+00	 1.1934627e-01	 1.5703502e+00	 1.1195486e-01	  1.2539071e+00 	 1.7358327e-01


.. parsed-literal::

     117	 1.5194452e+00	 1.1922499e-01	 1.5724464e+00	 1.1186875e-01	  1.2483952e+00 	 1.8026328e-01


.. parsed-literal::

     118	 1.5208738e+00	 1.1914495e-01	 1.5739865e+00	 1.1170359e-01	  1.2451158e+00 	 3.1928515e-01
     119	 1.5227903e+00	 1.1914905e-01	 1.5759673e+00	 1.1181411e-01	  1.2426648e+00 	 1.9837976e-01


.. parsed-literal::

     120	 1.5242083e+00	 1.1916492e-01	 1.5773758e+00	 1.1193925e-01	  1.2434934e+00 	 2.0839190e-01


.. parsed-literal::

     121	 1.5262470e+00	 1.1918305e-01	 1.5794010e+00	 1.1223323e-01	  1.2425397e+00 	 2.1402431e-01
     122	 1.5279241e+00	 1.1913912e-01	 1.5810116e+00	 1.1235838e-01	  1.2479363e+00 	 1.9704556e-01


.. parsed-literal::

     123	 1.5291776e+00	 1.1904286e-01	 1.5822188e+00	 1.1226911e-01	  1.2496972e+00 	 2.2300076e-01


.. parsed-literal::

     124	 1.5309730e+00	 1.1885811e-01	 1.5840152e+00	 1.1213814e-01	  1.2506854e+00 	 2.0583200e-01


.. parsed-literal::

     125	 1.5320605e+00	 1.1879148e-01	 1.5851276e+00	 1.1207384e-01	  1.2510544e+00 	 2.1442986e-01
     126	 1.5332891e+00	 1.1879044e-01	 1.5863810e+00	 1.1204927e-01	  1.2507226e+00 	 1.9279885e-01


.. parsed-literal::

     127	 1.5346877e+00	 1.1884137e-01	 1.5878447e+00	 1.1209041e-01	  1.2458474e+00 	 2.0003986e-01
     128	 1.5359909e+00	 1.1895019e-01	 1.5891947e+00	 1.1204830e-01	  1.2428882e+00 	 1.8581223e-01


.. parsed-literal::

     129	 1.5375043e+00	 1.1895450e-01	 1.5907523e+00	 1.1198127e-01	  1.2398089e+00 	 1.9383550e-01


.. parsed-literal::

     130	 1.5389258e+00	 1.1889105e-01	 1.5921882e+00	 1.1188576e-01	  1.2356416e+00 	 2.0375371e-01
     131	 1.5399136e+00	 1.1879287e-01	 1.5932163e+00	 1.1163915e-01	  1.2405445e+00 	 1.9057846e-01


.. parsed-literal::

     132	 1.5410912e+00	 1.1870875e-01	 1.5943514e+00	 1.1163951e-01	  1.2397517e+00 	 2.2381973e-01
     133	 1.5421234e+00	 1.1861839e-01	 1.5953634e+00	 1.1162631e-01	  1.2399113e+00 	 1.7196846e-01


.. parsed-literal::

     134	 1.5431524e+00	 1.1854996e-01	 1.5963931e+00	 1.1162400e-01	  1.2388122e+00 	 2.0196319e-01


.. parsed-literal::

     135	 1.5443024e+00	 1.1842790e-01	 1.5976196e+00	 1.1157625e-01	  1.2390353e+00 	 2.0414376e-01
     136	 1.5458767e+00	 1.1838804e-01	 1.5991936e+00	 1.1158772e-01	  1.2374244e+00 	 1.9857097e-01


.. parsed-literal::

     137	 1.5469022e+00	 1.1833148e-01	 1.6002499e+00	 1.1151655e-01	  1.2348771e+00 	 2.2191286e-01


.. parsed-literal::

     138	 1.5480293e+00	 1.1827615e-01	 1.6014631e+00	 1.1139885e-01	  1.2308411e+00 	 2.2023273e-01


.. parsed-literal::

     139	 1.5487788e+00	 1.1818553e-01	 1.6023565e+00	 1.1124804e-01	  1.2236247e+00 	 2.1088457e-01
     140	 1.5497306e+00	 1.1817857e-01	 1.6032814e+00	 1.1122175e-01	  1.2234848e+00 	 1.9925141e-01


.. parsed-literal::

     141	 1.5503244e+00	 1.1814423e-01	 1.6038869e+00	 1.1119285e-01	  1.2235353e+00 	 1.8769240e-01
     142	 1.5510036e+00	 1.1810621e-01	 1.6046092e+00	 1.1117054e-01	  1.2226009e+00 	 1.8416667e-01


.. parsed-literal::

     143	 1.5525475e+00	 1.1802519e-01	 1.6062269e+00	 1.1117630e-01	  1.2218934e+00 	 1.7709732e-01


.. parsed-literal::

     144	 1.5531075e+00	 1.1796235e-01	 1.6069538e+00	 1.1114025e-01	  1.2211138e+00 	 2.0691490e-01
     145	 1.5549750e+00	 1.1789960e-01	 1.6087707e+00	 1.1125861e-01	  1.2164829e+00 	 1.8890429e-01


.. parsed-literal::

     146	 1.5556290e+00	 1.1785784e-01	 1.6093609e+00	 1.1121403e-01	  1.2185101e+00 	 1.9125819e-01
     147	 1.5564884e+00	 1.1780177e-01	 1.6102169e+00	 1.1120258e-01	  1.2187049e+00 	 1.8383646e-01


.. parsed-literal::

     148	 1.5572513e+00	 1.1769334e-01	 1.6110339e+00	 1.1125106e-01	  1.2166925e+00 	 2.0772648e-01


.. parsed-literal::

     149	 1.5583343e+00	 1.1763840e-01	 1.6121122e+00	 1.1125311e-01	  1.2166048e+00 	 2.1500063e-01
     150	 1.5592789e+00	 1.1756678e-01	 1.6130999e+00	 1.1128011e-01	  1.2145827e+00 	 1.8394303e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.32 s, total: 2min 7s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd038b75300>



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
    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.24 s, sys: 43 ms, total: 2.28 s
    Wall time: 742 ms


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

