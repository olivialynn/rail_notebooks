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
       1	-3.4858933e-01	 3.2189841e-01	-3.3891468e-01	 3.1584606e-01	[-3.2672965e-01]	 4.5564771e-01


.. parsed-literal::

       2	-2.7659865e-01	 3.1107457e-01	-2.5300034e-01	 3.0533737e-01	[-2.3304012e-01]	 2.2769523e-01


.. parsed-literal::

       3	-2.3133511e-01	 2.9024601e-01	-1.8981171e-01	 2.8570234e-01	[-1.6408366e-01]	 2.8135395e-01
       4	-2.0137039e-01	 2.6631602e-01	-1.6106635e-01	 2.5957017e-01	[-1.2113805e-01]	 1.8994999e-01


.. parsed-literal::

       5	-1.0984485e-01	 2.5849729e-01	-7.5115960e-02	 2.5996555e-01	[-6.4871906e-02]	 2.1213651e-01


.. parsed-literal::

       6	-7.6144425e-02	 2.5260498e-01	-4.4298841e-02	 2.4793500e-01	[-2.6999870e-02]	 2.0934534e-01
       7	-5.8446302e-02	 2.5007915e-01	-3.3739411e-02	 2.4674502e-01	[-1.7663401e-02]	 1.7876720e-01


.. parsed-literal::

       8	-4.4979649e-02	 2.4783221e-01	-2.4416435e-02	 2.4481667e-01	[-7.4607054e-03]	 2.0169520e-01


.. parsed-literal::

       9	-2.9560514e-02	 2.4494474e-01	-1.2209298e-02	 2.4239540e-01	[ 4.8647846e-03]	 2.0897555e-01
      10	-2.1468672e-02	 2.4374015e-01	-6.5814702e-03	 2.4354762e-01	  4.0654746e-03 	 2.0517826e-01


.. parsed-literal::

      11	-1.5269854e-02	 2.4240832e-01	-9.7117918e-04	 2.4119554e-01	[ 1.1850600e-02]	 1.8163347e-01
      12	-1.3186809e-02	 2.4204591e-01	 8.3586900e-04	 2.4028444e-01	[ 1.5530799e-02]	 1.9049931e-01


.. parsed-literal::

      13	-8.7268428e-03	 2.4126720e-01	 4.9579023e-03	 2.3818962e-01	[ 2.3246957e-02]	 1.7736125e-01


.. parsed-literal::

      14	 1.5935115e-01	 2.2679590e-01	 1.8486874e-01	 2.1721264e-01	[ 2.2320520e-01]	 4.3325114e-01


.. parsed-literal::

      15	 2.1119305e-01	 2.2359509e-01	 2.4024403e-01	 2.1822072e-01	[ 2.6356408e-01]	 3.0936790e-01


.. parsed-literal::

      16	 2.6662037e-01	 2.1432443e-01	 2.9991820e-01	 2.1012554e-01	[ 3.2135674e-01]	 2.0380569e-01


.. parsed-literal::

      17	 3.3722338e-01	 2.0802346e-01	 3.6991962e-01	 2.0296503e-01	[ 3.9458108e-01]	 2.0344973e-01


.. parsed-literal::

      18	 3.8263534e-01	 2.0534640e-01	 4.1624126e-01	 1.9939473e-01	[ 4.5423186e-01]	 2.1611118e-01


.. parsed-literal::

      19	 4.3977871e-01	 2.0098025e-01	 4.7431309e-01	 1.9495468e-01	[ 5.1223922e-01]	 2.1485305e-01
      20	 5.4630037e-01	 1.9892655e-01	 5.8353158e-01	 1.9492176e-01	[ 6.0615167e-01]	 1.8814826e-01


.. parsed-literal::

      21	 5.9800813e-01	 1.9808840e-01	 6.3541390e-01	 1.9715897e-01	[ 6.3994629e-01]	 1.7473149e-01


.. parsed-literal::

      22	 6.4033904e-01	 1.9364761e-01	 6.7957487e-01	 1.9053046e-01	[ 6.8481134e-01]	 2.1204472e-01


.. parsed-literal::

      23	 6.8112722e-01	 1.9161677e-01	 7.1935960e-01	 1.8892247e-01	[ 7.2929210e-01]	 2.0174170e-01
      24	 7.2212223e-01	 1.8839151e-01	 7.6023086e-01	 1.8837817e-01	[ 7.5438742e-01]	 1.9943309e-01


.. parsed-literal::

      25	 7.4258920e-01	 1.9531751e-01	 7.8157500e-01	 1.9998355e-01	  7.5331143e-01 	 2.0829582e-01
      26	 7.9714550e-01	 1.9049033e-01	 8.3676130e-01	 1.9012619e-01	[ 8.1359969e-01]	 1.9339156e-01


.. parsed-literal::

      27	 8.3398398e-01	 1.8674699e-01	 8.7484142e-01	 1.8510525e-01	[ 8.5513571e-01]	 2.0606017e-01
      28	 8.7352749e-01	 1.8391493e-01	 9.1588600e-01	 1.8092622e-01	[ 8.9216231e-01]	 1.7014909e-01


.. parsed-literal::

      29	 9.0571490e-01	 1.8117266e-01	 9.4948908e-01	 1.7591214e-01	[ 9.3241597e-01]	 1.9899082e-01


.. parsed-literal::

      30	 9.1496843e-01	 1.7608902e-01	 9.5984818e-01	 1.7148218e-01	  9.1632040e-01 	 2.0634580e-01


.. parsed-literal::

      31	 9.5941384e-01	 1.7070175e-01	 1.0028467e+00	 1.6574084e-01	[ 9.7596768e-01]	 2.1695662e-01
      32	 9.8164648e-01	 1.6641881e-01	 1.0253341e+00	 1.6290803e-01	[ 9.9507082e-01]	 1.8386102e-01


.. parsed-literal::

      33	 1.0056472e+00	 1.6105066e-01	 1.0498231e+00	 1.5933940e-01	[ 1.0160820e+00]	 2.0688367e-01
      34	 1.0331292e+00	 1.5567362e-01	 1.0785792e+00	 1.5457424e-01	[ 1.0356318e+00]	 1.9841313e-01


.. parsed-literal::

      35	 1.0431793e+00	 1.5394008e-01	 1.0892812e+00	 1.5328088e-01	[ 1.0410873e+00]	 2.0068502e-01
      36	 1.0630322e+00	 1.5267474e-01	 1.1087541e+00	 1.5239013e-01	[ 1.0703107e+00]	 1.9078469e-01


.. parsed-literal::

      37	 1.0740173e+00	 1.5195461e-01	 1.1198263e+00	 1.5238450e-01	[ 1.0797376e+00]	 1.7021036e-01
      38	 1.0877549e+00	 1.5092476e-01	 1.1340493e+00	 1.5260647e-01	[ 1.0876898e+00]	 1.8454528e-01


.. parsed-literal::

      39	 1.1059500e+00	 1.4921591e-01	 1.1528059e+00	 1.5326506e-01	[ 1.0950832e+00]	 1.6373158e-01
      40	 1.1182489e+00	 1.4773494e-01	 1.1655855e+00	 1.5306986e-01	[ 1.0959453e+00]	 1.7914748e-01


.. parsed-literal::

      41	 1.1277469e+00	 1.4705044e-01	 1.1748713e+00	 1.5160942e-01	[ 1.1092589e+00]	 1.8730164e-01


.. parsed-literal::

      42	 1.1382354e+00	 1.4482946e-01	 1.1853901e+00	 1.5050045e-01	[ 1.1150838e+00]	 2.0194387e-01
      43	 1.1473089e+00	 1.4269977e-01	 1.1945671e+00	 1.4935690e-01	[ 1.1175138e+00]	 1.7887354e-01


.. parsed-literal::

      44	 1.1556799e+00	 1.4093842e-01	 1.2029833e+00	 1.5005399e-01	  1.1154623e+00 	 2.0191479e-01
      45	 1.1610406e+00	 1.4047847e-01	 1.2082384e+00	 1.4974695e-01	[ 1.1201223e+00]	 2.0146990e-01


.. parsed-literal::

      46	 1.1743141e+00	 1.3931479e-01	 1.2217634e+00	 1.4966737e-01	[ 1.1246334e+00]	 2.0778346e-01


.. parsed-literal::

      47	 1.1856336e+00	 1.3812316e-01	 1.2332711e+00	 1.4973785e-01	[ 1.1292698e+00]	 2.0707297e-01


.. parsed-literal::

      48	 1.2023919e+00	 1.3623683e-01	 1.2513340e+00	 1.5015999e-01	  1.1230470e+00 	 2.0818853e-01
      49	 1.2140581e+00	 1.3455206e-01	 1.2632448e+00	 1.4915038e-01	  1.1271315e+00 	 1.7853022e-01


.. parsed-literal::

      50	 1.2237688e+00	 1.3407729e-01	 1.2726307e+00	 1.4832625e-01	[ 1.1398608e+00]	 2.0207071e-01
      51	 1.2338144e+00	 1.3347551e-01	 1.2827617e+00	 1.4692400e-01	[ 1.1528799e+00]	 1.6186023e-01


.. parsed-literal::

      52	 1.2436631e+00	 1.3260135e-01	 1.2928306e+00	 1.4581315e-01	[ 1.1598070e+00]	 2.0403481e-01


.. parsed-literal::

      53	 1.2521534e+00	 1.3215549e-01	 1.3016759e+00	 1.4480963e-01	[ 1.1702865e+00]	 2.0836282e-01


.. parsed-literal::

      54	 1.2645378e+00	 1.3131426e-01	 1.3138795e+00	 1.4455449e-01	[ 1.1779609e+00]	 2.0852304e-01


.. parsed-literal::

      55	 1.2719323e+00	 1.3087100e-01	 1.3213071e+00	 1.4456178e-01	[ 1.1808034e+00]	 2.1108651e-01


.. parsed-literal::

      56	 1.2824530e+00	 1.3010940e-01	 1.3320791e+00	 1.4441699e-01	[ 1.1815282e+00]	 2.0733404e-01
      57	 1.2892794e+00	 1.3025166e-01	 1.3392015e+00	 1.4536816e-01	  1.1807843e+00 	 1.7852592e-01


.. parsed-literal::

      58	 1.2986334e+00	 1.2967226e-01	 1.3483177e+00	 1.4502115e-01	[ 1.1965603e+00]	 2.0601630e-01


.. parsed-literal::

      59	 1.3058903e+00	 1.2926370e-01	 1.3555260e+00	 1.4475526e-01	[ 1.2067047e+00]	 2.0547628e-01
      60	 1.3140222e+00	 1.2899464e-01	 1.3638697e+00	 1.4443952e-01	[ 1.2133229e+00]	 1.9370985e-01


.. parsed-literal::

      61	 1.3229011e+00	 1.2896302e-01	 1.3733308e+00	 1.4519202e-01	  1.2021617e+00 	 1.6642857e-01
      62	 1.3329518e+00	 1.2860125e-01	 1.3834105e+00	 1.4482042e-01	  1.2051081e+00 	 1.8674803e-01


.. parsed-literal::

      63	 1.3398291e+00	 1.2831916e-01	 1.3904554e+00	 1.4450182e-01	  1.1972780e+00 	 2.0957184e-01
      64	 1.3468930e+00	 1.2820452e-01	 1.3977806e+00	 1.4473987e-01	  1.1854831e+00 	 1.8487096e-01


.. parsed-literal::

      65	 1.3511558e+00	 1.2859827e-01	 1.4026235e+00	 1.4625410e-01	  1.1492173e+00 	 2.0674062e-01
      66	 1.3604546e+00	 1.2841021e-01	 1.4117432e+00	 1.4563694e-01	  1.1623438e+00 	 1.7106128e-01


.. parsed-literal::

      67	 1.3647582e+00	 1.2832377e-01	 1.4158841e+00	 1.4550862e-01	  1.1735905e+00 	 1.7334700e-01
      68	 1.3715809e+00	 1.2827686e-01	 1.4228814e+00	 1.4557442e-01	  1.1760471e+00 	 1.8523645e-01


.. parsed-literal::

      69	 1.3763642e+00	 1.2852207e-01	 1.4280361e+00	 1.4655349e-01	  1.1683594e+00 	 2.0469975e-01


.. parsed-literal::

      70	 1.3827471e+00	 1.2829962e-01	 1.4343524e+00	 1.4619717e-01	  1.1741712e+00 	 2.1446967e-01


.. parsed-literal::

      71	 1.3870376e+00	 1.2819815e-01	 1.4387463e+00	 1.4620768e-01	  1.1703482e+00 	 2.0986509e-01
      72	 1.3911561e+00	 1.2795542e-01	 1.4429516e+00	 1.4606227e-01	  1.1671175e+00 	 1.9716120e-01


.. parsed-literal::

      73	 1.3955862e+00	 1.2800578e-01	 1.4477971e+00	 1.4622589e-01	  1.1644788e+00 	 2.0163012e-01


.. parsed-literal::

      74	 1.4011021e+00	 1.2753440e-01	 1.4532238e+00	 1.4620991e-01	  1.1656663e+00 	 2.0567131e-01


.. parsed-literal::

      75	 1.4048220e+00	 1.2746810e-01	 1.4568967e+00	 1.4602468e-01	  1.1688921e+00 	 2.1549630e-01
      76	 1.4103601e+00	 1.2744739e-01	 1.4625863e+00	 1.4618128e-01	  1.1731508e+00 	 2.0003724e-01


.. parsed-literal::

      77	 1.4138188e+00	 1.2747170e-01	 1.4663841e+00	 1.4600682e-01	  1.1775595e+00 	 1.7739344e-01


.. parsed-literal::

      78	 1.4187215e+00	 1.2726830e-01	 1.4711797e+00	 1.4613474e-01	  1.1847855e+00 	 2.0968270e-01
      79	 1.4219217e+00	 1.2703962e-01	 1.4744372e+00	 1.4617806e-01	  1.1916985e+00 	 1.9400930e-01


.. parsed-literal::

      80	 1.4246946e+00	 1.2682815e-01	 1.4772701e+00	 1.4605988e-01	  1.1984460e+00 	 2.0736074e-01


.. parsed-literal::

      81	 1.4296156e+00	 1.2640994e-01	 1.4824937e+00	 1.4608252e-01	[ 1.2166962e+00]	 2.0472908e-01


.. parsed-literal::

      82	 1.4336312e+00	 1.2622758e-01	 1.4864990e+00	 1.4568267e-01	[ 1.2225998e+00]	 2.0567632e-01
      83	 1.4361659e+00	 1.2626976e-01	 1.4888966e+00	 1.4568615e-01	[ 1.2248409e+00]	 1.8122125e-01


.. parsed-literal::

      84	 1.4390144e+00	 1.2628944e-01	 1.4916862e+00	 1.4563473e-01	[ 1.2266801e+00]	 2.1359205e-01


.. parsed-literal::

      85	 1.4423061e+00	 1.2625373e-01	 1.4950169e+00	 1.4561775e-01	[ 1.2283386e+00]	 2.1348214e-01


.. parsed-literal::

      86	 1.4452509e+00	 1.2613776e-01	 1.4982176e+00	 1.4559705e-01	  1.2195996e+00 	 2.0794344e-01


.. parsed-literal::

      87	 1.4490186e+00	 1.2607169e-01	 1.5019597e+00	 1.4566460e-01	  1.2278046e+00 	 2.1419072e-01
      88	 1.4509675e+00	 1.2592738e-01	 1.5039108e+00	 1.4564176e-01	[ 1.2297234e+00]	 1.8957734e-01


.. parsed-literal::

      89	 1.4542325e+00	 1.2574819e-01	 1.5073184e+00	 1.4564970e-01	  1.2275853e+00 	 2.0979834e-01
      90	 1.4569100e+00	 1.2561194e-01	 1.5102271e+00	 1.4606545e-01	  1.2136000e+00 	 1.8952823e-01


.. parsed-literal::

      91	 1.4603011e+00	 1.2553465e-01	 1.5135966e+00	 1.4594169e-01	  1.2110882e+00 	 2.0334911e-01


.. parsed-literal::

      92	 1.4629903e+00	 1.2547274e-01	 1.5162420e+00	 1.4602629e-01	  1.2055250e+00 	 2.1344018e-01
      93	 1.4659901e+00	 1.2536327e-01	 1.5192551e+00	 1.4622284e-01	  1.1936174e+00 	 1.9927406e-01


.. parsed-literal::

      94	 1.4691740e+00	 1.2489245e-01	 1.5225490e+00	 1.4655953e-01	  1.1684508e+00 	 2.1918130e-01
      95	 1.4721434e+00	 1.2481698e-01	 1.5254241e+00	 1.4657339e-01	  1.1641409e+00 	 1.9934916e-01


.. parsed-literal::

      96	 1.4741190e+00	 1.2461721e-01	 1.5274023e+00	 1.4634329e-01	  1.1619323e+00 	 1.8753362e-01
      97	 1.4764787e+00	 1.2442907e-01	 1.5297779e+00	 1.4614784e-01	  1.1529840e+00 	 1.9969797e-01


.. parsed-literal::

      98	 1.4792397e+00	 1.2417650e-01	 1.5325772e+00	 1.4577512e-01	  1.1447865e+00 	 1.9575667e-01


.. parsed-literal::

      99	 1.4813783e+00	 1.2410930e-01	 1.5347166e+00	 1.4574900e-01	  1.1385870e+00 	 2.0631933e-01
     100	 1.4835998e+00	 1.2400157e-01	 1.5369323e+00	 1.4578218e-01	  1.1285165e+00 	 1.7636824e-01


.. parsed-literal::

     101	 1.4850601e+00	 1.2394883e-01	 1.5384322e+00	 1.4582050e-01	  1.1287443e+00 	 2.1222782e-01


.. parsed-literal::

     102	 1.4868145e+00	 1.2385999e-01	 1.5401329e+00	 1.4578611e-01	  1.1251709e+00 	 2.1178102e-01


.. parsed-literal::

     103	 1.4890896e+00	 1.2366897e-01	 1.5424025e+00	 1.4570283e-01	  1.1172749e+00 	 2.0090556e-01


.. parsed-literal::

     104	 1.4909037e+00	 1.2353506e-01	 1.5442430e+00	 1.4571074e-01	  1.1114554e+00 	 2.1410275e-01


.. parsed-literal::

     105	 1.4944197e+00	 1.2317271e-01	 1.5479528e+00	 1.4607712e-01	  1.0912876e+00 	 2.0585132e-01


.. parsed-literal::

     106	 1.4959291e+00	 1.2324234e-01	 1.5496649e+00	 1.4638611e-01	  1.0911063e+00 	 2.0713997e-01


.. parsed-literal::

     107	 1.4979742e+00	 1.2312172e-01	 1.5515710e+00	 1.4626886e-01	  1.0933248e+00 	 2.1566606e-01
     108	 1.4987884e+00	 1.2306700e-01	 1.5523933e+00	 1.4626711e-01	  1.0921054e+00 	 1.9961214e-01


.. parsed-literal::

     109	 1.5005293e+00	 1.2291263e-01	 1.5542514e+00	 1.4642673e-01	  1.0834294e+00 	 2.0673752e-01
     110	 1.5016940e+00	 1.2271169e-01	 1.5555965e+00	 1.4647943e-01	  1.0718157e+00 	 1.8406630e-01


.. parsed-literal::

     111	 1.5032750e+00	 1.2270393e-01	 1.5571363e+00	 1.4660926e-01	  1.0690741e+00 	 1.9680762e-01


.. parsed-literal::

     112	 1.5045477e+00	 1.2262632e-01	 1.5584337e+00	 1.4676903e-01	  1.0617769e+00 	 2.1414471e-01


.. parsed-literal::

     113	 1.5057076e+00	 1.2253786e-01	 1.5596155e+00	 1.4686831e-01	  1.0545990e+00 	 2.0816231e-01


.. parsed-literal::

     114	 1.5080268e+00	 1.2234219e-01	 1.5620396e+00	 1.4704887e-01	  1.0409096e+00 	 2.1128321e-01


.. parsed-literal::

     115	 1.5093820e+00	 1.2221078e-01	 1.5634293e+00	 1.4714454e-01	  1.0228263e+00 	 3.2507825e-01
     116	 1.5106495e+00	 1.2217552e-01	 1.5646843e+00	 1.4716649e-01	  1.0206117e+00 	 1.8671346e-01


.. parsed-literal::

     117	 1.5120273e+00	 1.2217140e-01	 1.5660480e+00	 1.4709272e-01	  1.0183065e+00 	 2.1640682e-01
     118	 1.5133577e+00	 1.2218646e-01	 1.5673909e+00	 1.4709578e-01	  1.0170994e+00 	 1.9754672e-01


.. parsed-literal::

     119	 1.5149683e+00	 1.2217124e-01	 1.5690479e+00	 1.4698725e-01	  1.0140323e+00 	 2.1006846e-01
     120	 1.5160676e+00	 1.2224216e-01	 1.5701751e+00	 1.4707962e-01	  1.0141025e+00 	 2.0349717e-01


.. parsed-literal::

     121	 1.5170367e+00	 1.2214633e-01	 1.5711453e+00	 1.4709551e-01	  1.0146794e+00 	 2.0557880e-01


.. parsed-literal::

     122	 1.5187237e+00	 1.2199648e-01	 1.5728787e+00	 1.4711000e-01	  1.0158271e+00 	 2.0750666e-01


.. parsed-literal::

     123	 1.5195286e+00	 1.2183462e-01	 1.5737472e+00	 1.4713142e-01	  1.0091871e+00 	 2.1468520e-01


.. parsed-literal::

     124	 1.5208549e+00	 1.2186005e-01	 1.5750195e+00	 1.4711321e-01	  1.0120808e+00 	 2.1265602e-01
     125	 1.5216538e+00	 1.2190655e-01	 1.5757895e+00	 1.4709453e-01	  1.0135252e+00 	 1.8624210e-01


.. parsed-literal::

     126	 1.5224804e+00	 1.2192413e-01	 1.5765807e+00	 1.4707408e-01	  1.0164429e+00 	 2.1336627e-01


.. parsed-literal::

     127	 1.5245408e+00	 1.2186100e-01	 1.5785551e+00	 1.4699389e-01	  1.0272011e+00 	 2.0754695e-01


.. parsed-literal::

     128	 1.5254185e+00	 1.2182885e-01	 1.5793925e+00	 1.4689939e-01	  1.0365478e+00 	 2.9850531e-01


.. parsed-literal::

     129	 1.5268541e+00	 1.2168763e-01	 1.5807949e+00	 1.4684336e-01	  1.0457153e+00 	 2.1616840e-01


.. parsed-literal::

     130	 1.5281197e+00	 1.2151445e-01	 1.5820548e+00	 1.4680844e-01	  1.0533697e+00 	 2.0457506e-01


.. parsed-literal::

     131	 1.5291992e+00	 1.2124332e-01	 1.5832269e+00	 1.4667967e-01	  1.0558704e+00 	 2.0945454e-01


.. parsed-literal::

     132	 1.5307895e+00	 1.2117375e-01	 1.5847748e+00	 1.4678860e-01	  1.0572262e+00 	 2.1458840e-01
     133	 1.5319141e+00	 1.2112143e-01	 1.5859207e+00	 1.4687481e-01	  1.0532530e+00 	 1.9697428e-01


.. parsed-literal::

     134	 1.5330475e+00	 1.2105620e-01	 1.5871065e+00	 1.4696297e-01	  1.0480408e+00 	 2.0886803e-01


.. parsed-literal::

     135	 1.5347215e+00	 1.2091756e-01	 1.5888383e+00	 1.4701288e-01	  1.0432224e+00 	 2.0626688e-01


.. parsed-literal::

     136	 1.5355475e+00	 1.2093515e-01	 1.5897181e+00	 1.4708326e-01	  1.0393828e+00 	 2.8454089e-01
     137	 1.5368422e+00	 1.2080871e-01	 1.5910041e+00	 1.4705148e-01	  1.0405161e+00 	 1.8738127e-01


.. parsed-literal::

     138	 1.5379301e+00	 1.2073872e-01	 1.5920697e+00	 1.4694996e-01	  1.0388586e+00 	 1.7514896e-01
     139	 1.5393642e+00	 1.2064324e-01	 1.5934887e+00	 1.4683381e-01	  1.0379786e+00 	 1.8415403e-01


.. parsed-literal::

     140	 1.5403641e+00	 1.2056426e-01	 1.5944990e+00	 1.4668965e-01	  1.0289314e+00 	 2.1882892e-01


.. parsed-literal::

     141	 1.5411765e+00	 1.2057001e-01	 1.5952877e+00	 1.4671754e-01	  1.0289961e+00 	 2.1510172e-01


.. parsed-literal::

     142	 1.5420221e+00	 1.2056945e-01	 1.5961601e+00	 1.4681181e-01	  1.0235799e+00 	 2.0699382e-01
     143	 1.5427955e+00	 1.2056983e-01	 1.5969536e+00	 1.4689186e-01	  1.0165236e+00 	 1.9829917e-01


.. parsed-literal::

     144	 1.5446805e+00	 1.2056719e-01	 1.5988908e+00	 1.4717952e-01	  9.9685056e-01 	 2.1563029e-01


.. parsed-literal::

     145	 1.5456026e+00	 1.2055667e-01	 1.5998531e+00	 1.4710167e-01	  9.8070397e-01 	 3.3143783e-01
     146	 1.5464525e+00	 1.2054859e-01	 1.6006883e+00	 1.4716406e-01	  9.7766394e-01 	 1.7143655e-01


.. parsed-literal::

     147	 1.5475017e+00	 1.2053028e-01	 1.6017445e+00	 1.4721982e-01	  9.7217335e-01 	 2.1132445e-01
     148	 1.5484810e+00	 1.2055417e-01	 1.6027553e+00	 1.4724799e-01	  9.6331037e-01 	 2.0014620e-01


.. parsed-literal::

     149	 1.5495404e+00	 1.2042827e-01	 1.6039457e+00	 1.4745808e-01	  9.3587027e-01 	 1.7894173e-01
     150	 1.5507258e+00	 1.2044301e-01	 1.6051353e+00	 1.4743780e-01	  9.2583557e-01 	 1.8726134e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.05 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f3d0176aed0>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.82 s, sys: 60 ms, total: 1.88 s
    Wall time: 607 ms


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




.. image:: 06_GPz_files/06_GPz_16_1.png


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




.. image:: 06_GPz_files/06_GPz_19_1.png

