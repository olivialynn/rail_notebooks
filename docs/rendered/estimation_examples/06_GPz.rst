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
       1	-3.3965599e-01	 3.1864237e-01	-3.2997787e-01	 3.2782860e-01	[-3.4778564e-01]	 4.6739960e-01


.. parsed-literal::

       2	-2.6301648e-01	 3.0632929e-01	-2.3717365e-01	 3.1602671e-01	[-2.6757674e-01]	 2.3295927e-01


.. parsed-literal::

       3	-2.1698287e-01	 2.8547490e-01	-1.7286146e-01	 2.9729790e-01	[-2.2564224e-01]	 2.8556156e-01


.. parsed-literal::

       4	-1.7453734e-01	 2.6710162e-01	-1.2360824e-01	 2.8226669e-01	[-2.0293840e-01]	 2.9194641e-01


.. parsed-literal::

       5	-1.2707742e-01	 2.5452514e-01	-9.7126179e-02	 2.6744313e-01	[-1.7585164e-01]	 2.1222067e-01


.. parsed-literal::

       6	-6.2033835e-02	 2.5046003e-01	-3.4253336e-02	 2.6309030e-01	[-8.6483657e-02]	 2.1329093e-01
       7	-4.4678838e-02	 2.4673764e-01	-2.0492285e-02	 2.5951319e-01	[-7.0542229e-02]	 1.8216205e-01


.. parsed-literal::

       8	-3.3495714e-02	 2.4495226e-01	-1.2406899e-02	 2.5758199e-01	[-6.4201590e-02]	 2.0546865e-01
       9	-1.7177459e-02	 2.4192508e-01	 4.2150559e-04	 2.5424714e-01	[-5.0212510e-02]	 1.7349768e-01


.. parsed-literal::

      10	-6.4515005e-03	 2.3990802e-01	 9.0056889e-03	 2.5208525e-01	[-4.4902363e-02]	 1.8885279e-01
      11	-1.6119896e-03	 2.3915724e-01	 1.2974894e-02	 2.5064847e-01	[-3.6886237e-02]	 2.0380831e-01


.. parsed-literal::

      12	 2.0449242e-03	 2.3853067e-01	 1.6010965e-02	 2.5017153e-01	[-3.3248279e-02]	 2.0010734e-01


.. parsed-literal::

      13	 5.0926096e-03	 2.3795974e-01	 1.8903436e-02	 2.4977477e-01	[-3.1015764e-02]	 2.0290208e-01


.. parsed-literal::

      14	 1.0162154e-01	 2.2292063e-01	 1.2288286e-01	 2.3784741e-01	[ 9.2263951e-02]	 4.4185829e-01


.. parsed-literal::

      15	 1.9470542e-01	 2.1759942e-01	 2.1834375e-01	 2.3181173e-01	[ 1.7715790e-01]	 2.0717931e-01


.. parsed-literal::

      16	 2.4146656e-01	 2.1210599e-01	 2.7473354e-01	 2.2611916e-01	[ 2.0301973e-01]	 2.0303988e-01
      17	 3.0940259e-01	 2.1220375e-01	 3.4139381e-01	 2.2577158e-01	[ 2.8109807e-01]	 1.7716479e-01


.. parsed-literal::

      18	 3.3958651e-01	 2.1056638e-01	 3.7150826e-01	 2.2430819e-01	[ 3.1627083e-01]	 2.1244311e-01
      19	 3.7881536e-01	 2.0804771e-01	 4.1118625e-01	 2.2443544e-01	[ 3.5663488e-01]	 1.9386959e-01


.. parsed-literal::

      20	 4.4078784e-01	 2.0764728e-01	 4.7412028e-01	 2.2267833e-01	[ 4.3085770e-01]	 2.0185423e-01


.. parsed-literal::

      21	 5.2414017e-01	 2.0364594e-01	 5.6005346e-01	 2.1929213e-01	[ 5.1555701e-01]	 2.0657110e-01


.. parsed-literal::

      22	 5.7673027e-01	 2.0018809e-01	 6.1607685e-01	 2.1258201e-01	[ 5.9419476e-01]	 2.1582341e-01


.. parsed-literal::

      23	 6.2373731e-01	 1.9726083e-01	 6.6261363e-01	 2.0979162e-01	[ 6.4587831e-01]	 2.1335793e-01


.. parsed-literal::

      24	 6.6537100e-01	 1.9250992e-01	 7.0393063e-01	 2.0563817e-01	[ 6.9003608e-01]	 2.0841217e-01


.. parsed-literal::

      25	 7.1051195e-01	 1.9017076e-01	 7.5012745e-01	 2.0432945e-01	[ 7.4891959e-01]	 2.0956588e-01


.. parsed-literal::

      26	 7.3647201e-01	 1.9210832e-01	 7.7548861e-01	 2.0716439e-01	[ 7.6520986e-01]	 2.0994782e-01


.. parsed-literal::

      27	 7.6976963e-01	 1.9074300e-01	 8.0776382e-01	 2.0713464e-01	[ 7.9607967e-01]	 2.1376300e-01


.. parsed-literal::

      28	 7.9373118e-01	 1.9017221e-01	 8.3385252e-01	 2.0533204e-01	[ 8.2566693e-01]	 2.1660614e-01


.. parsed-literal::

      29	 8.1627400e-01	 1.9071099e-01	 8.5626347e-01	 2.0353123e-01	[ 8.5695548e-01]	 2.2237229e-01
      30	 8.4280417e-01	 1.8903727e-01	 8.8282261e-01	 2.0161400e-01	[ 8.8521643e-01]	 1.9070816e-01


.. parsed-literal::

      31	 8.7183112e-01	 1.8708043e-01	 9.1231325e-01	 2.0029946e-01	[ 9.0697608e-01]	 1.7899680e-01
      32	 8.9352162e-01	 1.8414731e-01	 9.3442189e-01	 1.9818326e-01	[ 9.1933585e-01]	 1.9354987e-01


.. parsed-literal::

      33	 9.1166020e-01	 1.8297225e-01	 9.5288429e-01	 1.9764896e-01	[ 9.3006038e-01]	 2.1065307e-01


.. parsed-literal::

      34	 9.2920900e-01	 1.7921789e-01	 9.7147966e-01	 1.9307961e-01	[ 9.4231369e-01]	 2.1671152e-01


.. parsed-literal::

      35	 9.4685895e-01	 1.7761162e-01	 9.8977213e-01	 1.9309377e-01	[ 9.5996227e-01]	 2.1536064e-01
      36	 9.6379521e-01	 1.7600254e-01	 1.0068884e+00	 1.9173989e-01	[ 9.7485262e-01]	 2.0752048e-01


.. parsed-literal::

      37	 9.7929214e-01	 1.7377162e-01	 1.0228371e+00	 1.8987764e-01	[ 9.8891220e-01]	 2.2192621e-01


.. parsed-literal::

      38	 1.0018647e+00	 1.6905083e-01	 1.0468018e+00	 1.8493167e-01	[ 1.0082439e+00]	 2.2161937e-01
      39	 1.0247436e+00	 1.6635536e-01	 1.0703076e+00	 1.8256453e-01	[ 1.0282751e+00]	 1.9122076e-01


.. parsed-literal::

      40	 1.0368214e+00	 1.6505471e-01	 1.0825529e+00	 1.8096889e-01	[ 1.0395285e+00]	 2.1787739e-01


.. parsed-literal::

      41	 1.0519566e+00	 1.6245025e-01	 1.0985240e+00	 1.7768798e-01	[ 1.0492105e+00]	 2.1502066e-01
      42	 1.0643668e+00	 1.6138436e-01	 1.1109692e+00	 1.7657111e-01	[ 1.0595800e+00]	 1.9250894e-01


.. parsed-literal::

      43	 1.0739008e+00	 1.6040181e-01	 1.1203649e+00	 1.7550844e-01	[ 1.0682166e+00]	 2.1670341e-01


.. parsed-literal::

      44	 1.0893344e+00	 1.5713196e-01	 1.1363684e+00	 1.7206833e-01	[ 1.0763293e+00]	 2.2021365e-01


.. parsed-literal::

      45	 1.1007367e+00	 1.5610747e-01	 1.1476845e+00	 1.7176540e-01	[ 1.0858892e+00]	 2.0962310e-01


.. parsed-literal::

      46	 1.1128987e+00	 1.5435969e-01	 1.1599749e+00	 1.7020285e-01	[ 1.0959103e+00]	 2.1502829e-01


.. parsed-literal::

      47	 1.1236554e+00	 1.5234904e-01	 1.1710177e+00	 1.6802260e-01	[ 1.1046633e+00]	 2.1043682e-01


.. parsed-literal::

      48	 1.1346466e+00	 1.5158157e-01	 1.1821692e+00	 1.6676029e-01	[ 1.1182910e+00]	 2.1140742e-01
      49	 1.1482910e+00	 1.5026032e-01	 1.1963356e+00	 1.6455222e-01	[ 1.1336598e+00]	 2.0052004e-01


.. parsed-literal::

      50	 1.1574810e+00	 1.4964270e-01	 1.2053530e+00	 1.6435115e-01	[ 1.1432976e+00]	 2.0755744e-01
      51	 1.1644279e+00	 1.4892672e-01	 1.2124013e+00	 1.6376319e-01	[ 1.1484408e+00]	 1.7556143e-01


.. parsed-literal::

      52	 1.1777283e+00	 1.4829094e-01	 1.2261578e+00	 1.6362737e-01	[ 1.1541978e+00]	 1.8461132e-01
      53	 1.1860132e+00	 1.4759351e-01	 1.2347030e+00	 1.6274425e-01	[ 1.1594720e+00]	 1.9852567e-01


.. parsed-literal::

      54	 1.1931625e+00	 1.4705165e-01	 1.2416625e+00	 1.6234973e-01	[ 1.1656665e+00]	 2.0805430e-01


.. parsed-literal::

      55	 1.2019843e+00	 1.4620250e-01	 1.2506162e+00	 1.6159917e-01	[ 1.1727007e+00]	 2.1124649e-01


.. parsed-literal::

      56	 1.2082192e+00	 1.4548884e-01	 1.2568581e+00	 1.6061183e-01	[ 1.1783525e+00]	 2.1314931e-01


.. parsed-literal::

      57	 1.2176062e+00	 1.4486492e-01	 1.2668071e+00	 1.5952264e-01	[ 1.1849363e+00]	 2.0335627e-01


.. parsed-literal::

      58	 1.2272985e+00	 1.4360814e-01	 1.2765083e+00	 1.5810095e-01	[ 1.1905366e+00]	 2.2091556e-01


.. parsed-literal::

      59	 1.2332843e+00	 1.4319114e-01	 1.2824873e+00	 1.5782180e-01	[ 1.1950500e+00]	 2.1489429e-01
      60	 1.2453615e+00	 1.4225207e-01	 1.2950821e+00	 1.5767541e-01	[ 1.2011930e+00]	 1.9022274e-01


.. parsed-literal::

      61	 1.2515377e+00	 1.4085822e-01	 1.3014776e+00	 1.5712046e-01	[ 1.2018675e+00]	 2.0434237e-01


.. parsed-literal::

      62	 1.2590633e+00	 1.4041192e-01	 1.3088356e+00	 1.5674378e-01	[ 1.2085848e+00]	 2.2088385e-01
      63	 1.2702854e+00	 1.3892737e-01	 1.3203324e+00	 1.5585249e-01	[ 1.2126110e+00]	 1.7124557e-01


.. parsed-literal::

      64	 1.2764777e+00	 1.3838762e-01	 1.3265292e+00	 1.5546858e-01	  1.2100561e+00 	 1.8626356e-01
      65	 1.2846254e+00	 1.3734720e-01	 1.3349360e+00	 1.5502688e-01	  1.2101529e+00 	 2.0044208e-01


.. parsed-literal::

      66	 1.2923348e+00	 1.3638425e-01	 1.3429202e+00	 1.5449862e-01	  1.2122367e+00 	 1.9925785e-01


.. parsed-literal::

      67	 1.3000542e+00	 1.3547639e-01	 1.3507677e+00	 1.5381597e-01	[ 1.2178280e+00]	 2.1726274e-01


.. parsed-literal::

      68	 1.3074679e+00	 1.3472812e-01	 1.3582340e+00	 1.5283132e-01	[ 1.2266469e+00]	 2.2203994e-01
      69	 1.3145147e+00	 1.3414555e-01	 1.3652841e+00	 1.5216658e-01	[ 1.2363857e+00]	 1.8688822e-01


.. parsed-literal::

      70	 1.3211910e+00	 1.3338067e-01	 1.3720707e+00	 1.5139895e-01	[ 1.2466444e+00]	 2.1853280e-01
      71	 1.3285551e+00	 1.3359055e-01	 1.3792385e+00	 1.5208101e-01	[ 1.2541886e+00]	 1.8316698e-01


.. parsed-literal::

      72	 1.3336563e+00	 1.3352722e-01	 1.3844111e+00	 1.5246640e-01	[ 1.2570269e+00]	 2.0838189e-01


.. parsed-literal::

      73	 1.3421645e+00	 1.3345330e-01	 1.3931216e+00	 1.5309290e-01	[ 1.2600902e+00]	 2.0553732e-01


.. parsed-literal::

      74	 1.3472144e+00	 1.3300847e-01	 1.3984397e+00	 1.5295449e-01	[ 1.2613671e+00]	 2.8870988e-01


.. parsed-literal::

      75	 1.3535910e+00	 1.3245898e-01	 1.4049683e+00	 1.5238553e-01	[ 1.2621248e+00]	 2.1420383e-01
      76	 1.3610456e+00	 1.3199775e-01	 1.4125287e+00	 1.5156008e-01	[ 1.2630988e+00]	 1.9808483e-01


.. parsed-literal::

      77	 1.3665829e+00	 1.3148376e-01	 1.4182800e+00	 1.5116522e-01	[ 1.2659690e+00]	 2.0213008e-01
      78	 1.3740452e+00	 1.3112129e-01	 1.4257699e+00	 1.5047303e-01	[ 1.2701870e+00]	 1.9491291e-01


.. parsed-literal::

      79	 1.3819394e+00	 1.3091087e-01	 1.4338108e+00	 1.5063735e-01	[ 1.2773622e+00]	 2.1492100e-01
      80	 1.3870635e+00	 1.3055363e-01	 1.4389709e+00	 1.5044431e-01	[ 1.2786389e+00]	 1.6390347e-01


.. parsed-literal::

      81	 1.3906259e+00	 1.3004729e-01	 1.4424691e+00	 1.5001130e-01	[ 1.2802977e+00]	 2.1128607e-01


.. parsed-literal::

      82	 1.3949701e+00	 1.2946775e-01	 1.4469030e+00	 1.4990369e-01	  1.2772682e+00 	 2.1284676e-01


.. parsed-literal::

      83	 1.3999873e+00	 1.2880987e-01	 1.4519944e+00	 1.4979653e-01	  1.2765783e+00 	 2.1274781e-01


.. parsed-literal::

      84	 1.4061591e+00	 1.2819523e-01	 1.4582968e+00	 1.4983955e-01	  1.2734354e+00 	 2.1506333e-01
      85	 1.4117679e+00	 1.2774201e-01	 1.4639458e+00	 1.4974183e-01	  1.2768284e+00 	 1.8581510e-01


.. parsed-literal::

      86	 1.4170755e+00	 1.2739301e-01	 1.4693169e+00	 1.4962831e-01	[ 1.2819343e+00]	 1.8717837e-01
      87	 1.4213760e+00	 1.2707504e-01	 1.4737456e+00	 1.4943962e-01	[ 1.2870921e+00]	 1.7940140e-01


.. parsed-literal::

      88	 1.4254768e+00	 1.2672839e-01	 1.4779191e+00	 1.4919078e-01	[ 1.2894884e+00]	 2.0091081e-01
      89	 1.4297010e+00	 1.2611948e-01	 1.4824186e+00	 1.4878432e-01	  1.2880318e+00 	 1.8981600e-01


.. parsed-literal::

      90	 1.4335443e+00	 1.2595734e-01	 1.4862202e+00	 1.4860238e-01	  1.2868749e+00 	 2.1464992e-01


.. parsed-literal::

      91	 1.4359327e+00	 1.2582147e-01	 1.4885494e+00	 1.4862795e-01	  1.2878815e+00 	 2.0786619e-01


.. parsed-literal::

      92	 1.4409039e+00	 1.2526757e-01	 1.4935756e+00	 1.4835178e-01	  1.2859137e+00 	 2.0724273e-01


.. parsed-literal::

      93	 1.4421015e+00	 1.2487315e-01	 1.4949976e+00	 1.4874745e-01	  1.2869190e+00 	 2.1051598e-01


.. parsed-literal::

      94	 1.4457796e+00	 1.2473353e-01	 1.4985332e+00	 1.4832249e-01	  1.2888944e+00 	 2.1709514e-01


.. parsed-literal::

      95	 1.4475922e+00	 1.2458635e-01	 1.5003578e+00	 1.4814602e-01	  1.2889391e+00 	 2.0715690e-01
      96	 1.4503545e+00	 1.2430102e-01	 1.5031549e+00	 1.4794486e-01	[ 1.2896037e+00]	 1.8664527e-01


.. parsed-literal::

      97	 1.4534368e+00	 1.2389798e-01	 1.5063352e+00	 1.4782312e-01	  1.2867699e+00 	 1.8249893e-01


.. parsed-literal::

      98	 1.4570861e+00	 1.2360615e-01	 1.5099658e+00	 1.4770957e-01	[ 1.2908084e+00]	 2.0709038e-01


.. parsed-literal::

      99	 1.4600540e+00	 1.2337760e-01	 1.5129402e+00	 1.4771945e-01	[ 1.2921082e+00]	 2.0270085e-01


.. parsed-literal::

     100	 1.4635782e+00	 1.2310005e-01	 1.5164402e+00	 1.4780435e-01	[ 1.2940339e+00]	 2.0558524e-01
     101	 1.4658195e+00	 1.2267254e-01	 1.5187336e+00	 1.4789379e-01	[ 1.2948930e+00]	 1.9842076e-01


.. parsed-literal::

     102	 1.4695400e+00	 1.2251932e-01	 1.5223761e+00	 1.4804283e-01	[ 1.2957194e+00]	 2.1650243e-01


.. parsed-literal::

     103	 1.4716526e+00	 1.2237087e-01	 1.5244897e+00	 1.4807733e-01	[ 1.2968768e+00]	 2.1345496e-01


.. parsed-literal::

     104	 1.4740660e+00	 1.2217757e-01	 1.5269795e+00	 1.4819927e-01	  1.2947565e+00 	 2.0200968e-01


.. parsed-literal::

     105	 1.4769160e+00	 1.2195152e-01	 1.5300077e+00	 1.4839315e-01	  1.2923322e+00 	 2.1021700e-01
     106	 1.4795750e+00	 1.2189837e-01	 1.5327239e+00	 1.4847137e-01	  1.2892510e+00 	 1.9955397e-01


.. parsed-literal::

     107	 1.4826752e+00	 1.2176259e-01	 1.5359078e+00	 1.4866060e-01	  1.2817776e+00 	 1.8541431e-01


.. parsed-literal::

     108	 1.4853703e+00	 1.2179828e-01	 1.5386630e+00	 1.4875668e-01	  1.2796632e+00 	 2.0392370e-01


.. parsed-literal::

     109	 1.4875671e+00	 1.2174592e-01	 1.5408667e+00	 1.4886504e-01	  1.2704918e+00 	 2.0927596e-01
     110	 1.4894695e+00	 1.2171602e-01	 1.5426875e+00	 1.4883258e-01	  1.2744982e+00 	 1.8093777e-01


.. parsed-literal::

     111	 1.4920459e+00	 1.2161601e-01	 1.5452309e+00	 1.4869663e-01	  1.2788741e+00 	 2.0452929e-01
     112	 1.4947415e+00	 1.2151585e-01	 1.5479611e+00	 1.4851421e-01	  1.2799515e+00 	 1.7941236e-01


.. parsed-literal::

     113	 1.4983780e+00	 1.2130598e-01	 1.5517383e+00	 1.4817325e-01	  1.2793057e+00 	 2.0658493e-01


.. parsed-literal::

     114	 1.5009120e+00	 1.2134242e-01	 1.5544756e+00	 1.4826091e-01	  1.2715510e+00 	 2.1191430e-01
     115	 1.5033548e+00	 1.2113085e-01	 1.5568124e+00	 1.4801723e-01	  1.2758657e+00 	 2.0481753e-01


.. parsed-literal::

     116	 1.5055952e+00	 1.2099151e-01	 1.5590593e+00	 1.4788994e-01	  1.2744484e+00 	 2.0371485e-01
     117	 1.5075229e+00	 1.2085618e-01	 1.5610404e+00	 1.4776219e-01	  1.2715261e+00 	 2.0009875e-01


.. parsed-literal::

     118	 1.5087081e+00	 1.2063203e-01	 1.5623207e+00	 1.4763132e-01	  1.2680962e+00 	 2.0390248e-01


.. parsed-literal::

     119	 1.5117780e+00	 1.2051417e-01	 1.5653731e+00	 1.4757035e-01	  1.2674647e+00 	 2.0755696e-01
     120	 1.5134364e+00	 1.2040152e-01	 1.5670439e+00	 1.4756039e-01	  1.2676348e+00 	 1.8663073e-01


.. parsed-literal::

     121	 1.5154220e+00	 1.2015055e-01	 1.5690722e+00	 1.4753548e-01	  1.2651047e+00 	 2.0468998e-01


.. parsed-literal::

     122	 1.5167643e+00	 1.2001559e-01	 1.5704845e+00	 1.4767218e-01	  1.2643076e+00 	 3.2709074e-01
     123	 1.5186887e+00	 1.1979131e-01	 1.5724097e+00	 1.4770866e-01	  1.2606222e+00 	 1.7643380e-01


.. parsed-literal::

     124	 1.5205191e+00	 1.1963658e-01	 1.5742384e+00	 1.4780260e-01	  1.2578035e+00 	 2.0160389e-01
     125	 1.5223574e+00	 1.1958005e-01	 1.5760682e+00	 1.4792272e-01	  1.2568146e+00 	 1.9747663e-01


.. parsed-literal::

     126	 1.5256091e+00	 1.1954884e-01	 1.5792947e+00	 1.4805997e-01	  1.2581589e+00 	 2.1456289e-01


.. parsed-literal::

     127	 1.5272182e+00	 1.1956407e-01	 1.5809315e+00	 1.4804507e-01	  1.2604494e+00 	 3.3369207e-01
     128	 1.5298843e+00	 1.1962143e-01	 1.5835694e+00	 1.4802506e-01	  1.2642785e+00 	 2.0110536e-01


.. parsed-literal::

     129	 1.5316309e+00	 1.1971592e-01	 1.5853072e+00	 1.4796253e-01	  1.2666053e+00 	 2.0879579e-01
     130	 1.5333451e+00	 1.1980039e-01	 1.5870637e+00	 1.4783404e-01	  1.2695307e+00 	 1.8337131e-01


.. parsed-literal::

     131	 1.5350303e+00	 1.1992020e-01	 1.5887650e+00	 1.4785062e-01	  1.2697632e+00 	 1.7352629e-01


.. parsed-literal::

     132	 1.5368909e+00	 1.1997677e-01	 1.5906743e+00	 1.4787208e-01	  1.2663588e+00 	 2.0781755e-01


.. parsed-literal::

     133	 1.5389090e+00	 1.2016186e-01	 1.5927587e+00	 1.4810818e-01	  1.2626738e+00 	 2.1177626e-01
     134	 1.5407489e+00	 1.2026230e-01	 1.5946438e+00	 1.4806585e-01	  1.2570586e+00 	 2.0011735e-01


.. parsed-literal::

     135	 1.5420416e+00	 1.2025922e-01	 1.5959257e+00	 1.4807099e-01	  1.2575632e+00 	 2.1842504e-01
     136	 1.5435711e+00	 1.2016985e-01	 1.5974507e+00	 1.4794730e-01	  1.2579673e+00 	 1.8266225e-01


.. parsed-literal::

     137	 1.5452233e+00	 1.1990832e-01	 1.5991581e+00	 1.4767106e-01	  1.2560147e+00 	 2.0957017e-01


.. parsed-literal::

     138	 1.5470919e+00	 1.1961557e-01	 1.6010450e+00	 1.4724536e-01	  1.2542272e+00 	 2.1220875e-01


.. parsed-literal::

     139	 1.5485695e+00	 1.1941900e-01	 1.6025497e+00	 1.4703851e-01	  1.2513052e+00 	 2.1232247e-01
     140	 1.5498606e+00	 1.1927015e-01	 1.6038655e+00	 1.4684391e-01	  1.2504113e+00 	 1.7652917e-01


.. parsed-literal::

     141	 1.5521002e+00	 1.1919172e-01	 1.6061570e+00	 1.4689129e-01	  1.2474728e+00 	 2.0047379e-01


.. parsed-literal::

     142	 1.5535836e+00	 1.1910426e-01	 1.6077684e+00	 1.4708363e-01	  1.2445940e+00 	 2.1669054e-01
     143	 1.5561625e+00	 1.1906599e-01	 1.6103010e+00	 1.4724889e-01	  1.2452407e+00 	 1.8871570e-01


.. parsed-literal::

     144	 1.5573854e+00	 1.1907394e-01	 1.6115199e+00	 1.4741283e-01	  1.2441639e+00 	 2.0419145e-01


.. parsed-literal::

     145	 1.5586398e+00	 1.1905420e-01	 1.6128071e+00	 1.4759259e-01	  1.2416569e+00 	 2.0813799e-01


.. parsed-literal::

     146	 1.5605292e+00	 1.1904461e-01	 1.6147567e+00	 1.4773696e-01	  1.2354494e+00 	 2.1014428e-01
     147	 1.5615577e+00	 1.1907138e-01	 1.6159107e+00	 1.4802037e-01	  1.2295036e+00 	 2.0127201e-01


.. parsed-literal::

     148	 1.5629395e+00	 1.1899734e-01	 1.6172579e+00	 1.4786376e-01	  1.2313975e+00 	 1.8022537e-01
     149	 1.5636175e+00	 1.1896954e-01	 1.6179309e+00	 1.4779824e-01	  1.2320688e+00 	 1.9439793e-01


.. parsed-literal::

     150	 1.5648841e+00	 1.1885247e-01	 1.6192430e+00	 1.4774279e-01	  1.2315308e+00 	 2.1216893e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.32 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f92d0f462f0>



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
    CPU times: user 1.84 s, sys: 36 ms, total: 1.87 s
    Wall time: 596 ms


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

