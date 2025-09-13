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
       1	-3.5600601e-01	 3.2416870e-01	-3.4631146e-01	 3.0514125e-01	[-3.1211939e-01]	 4.6222091e-01


.. parsed-literal::

       2	-2.8378393e-01	 3.1334614e-01	-2.5991765e-01	 2.9506408e-01	[-2.0689109e-01]	 2.2915554e-01


.. parsed-literal::

       3	-2.4077681e-01	 2.9347913e-01	-2.0018301e-01	 2.7373789e-01	[-1.2402386e-01]	 2.9049087e-01
       4	-2.0088300e-01	 2.6912427e-01	-1.5972919e-01	 2.4796602e-01	[-4.4484588e-02]	 1.8903971e-01


.. parsed-literal::

       5	-1.1618579e-01	 2.5992340e-01	-8.2551706e-02	 2.4154174e-01	[ 2.9904000e-04]	 1.9840336e-01


.. parsed-literal::

       6	-8.1334970e-02	 2.5442033e-01	-5.1258791e-02	 2.3871099e-01	[ 4.9937113e-03]	 2.0874691e-01


.. parsed-literal::

       7	-6.4581999e-02	 2.5175593e-01	-4.0151094e-02	 2.3709890e-01	[ 1.6227432e-02]	 2.0077443e-01


.. parsed-literal::

       8	-5.1623425e-02	 2.4958840e-01	-3.1223646e-02	 2.3546176e-01	[ 2.4723899e-02]	 2.0859957e-01


.. parsed-literal::

       9	-3.8570847e-02	 2.4717394e-01	-2.0995845e-02	 2.3349732e-01	[ 3.3353500e-02]	 2.1813107e-01
      10	-2.7181390e-02	 2.4487900e-01	-1.1812283e-02	 2.3218465e-01	[ 4.2662918e-02]	 1.8426871e-01


.. parsed-literal::

      11	-2.1734939e-02	 2.4414717e-01	-7.7790314e-03	 2.3107219e-01	  4.2449953e-02 	 2.0751619e-01


.. parsed-literal::

      12	-1.8741749e-02	 2.4366230e-01	-4.9744917e-03	 2.3062792e-01	[ 4.8052126e-02]	 2.1656799e-01
      13	-1.5262248e-02	 2.4294131e-01	-1.5864996e-03	 2.2949159e-01	[ 5.3890686e-02]	 1.7276645e-01


.. parsed-literal::

      14	-1.0330326e-02	 2.4185414e-01	 3.9938365e-03	 2.2846731e-01	[ 6.0587837e-02]	 2.1268344e-01


.. parsed-literal::

      15	 1.5309128e-01	 2.2415377e-01	 1.7583036e-01	 2.1346739e-01	[ 2.1245453e-01]	 3.0072951e-01
      16	 2.4128216e-01	 2.1861655e-01	 2.6731163e-01	 2.0760236e-01	[ 3.1498253e-01]	 1.9853950e-01


.. parsed-literal::

      17	 2.6003952e-01	 2.1320472e-01	 2.9429167e-01	 2.0224461e-01	[ 3.7303045e-01]	 2.5645661e-01


.. parsed-literal::

      18	 3.2796956e-01	 2.1310654e-01	 3.6026909e-01	 2.0521906e-01	[ 4.1731844e-01]	 2.1844268e-01


.. parsed-literal::

      19	 3.6266026e-01	 2.0821517e-01	 3.9398104e-01	 1.9894491e-01	[ 4.4716336e-01]	 2.0442748e-01


.. parsed-literal::

      20	 4.0661273e-01	 2.0499745e-01	 4.3838860e-01	 1.9542982e-01	[ 4.9016164e-01]	 2.1136451e-01
      21	 5.0365113e-01	 2.0565915e-01	 5.3690603e-01	 1.9325544e-01	[ 5.8601469e-01]	 1.7150617e-01


.. parsed-literal::

      22	 5.8140638e-01	 2.0728285e-01	 6.1997559e-01	 1.9394544e-01	[ 6.4596815e-01]	 2.0704889e-01


.. parsed-literal::

      23	 6.3528239e-01	 2.0095021e-01	 6.7349671e-01	 1.8761482e-01	[ 6.8267867e-01]	 2.0156384e-01


.. parsed-literal::

      24	 6.7278675e-01	 1.9651802e-01	 7.0872015e-01	 1.8391865e-01	[ 7.2580316e-01]	 2.0708251e-01


.. parsed-literal::

      25	 6.9908275e-01	 1.9658535e-01	 7.3490475e-01	 1.8483071e-01	[ 7.5691757e-01]	 2.1361899e-01


.. parsed-literal::

      26	 7.1851375e-01	 2.0106144e-01	 7.5448641e-01	 1.8742991e-01	[ 7.8278311e-01]	 2.1374369e-01


.. parsed-literal::

      27	 7.4687356e-01	 1.9770867e-01	 7.8395872e-01	 1.8395020e-01	[ 7.9948094e-01]	 2.1095228e-01


.. parsed-literal::

      28	 7.7030307e-01	 1.9643950e-01	 8.0791670e-01	 1.8299561e-01	[ 8.2307237e-01]	 2.1120763e-01


.. parsed-literal::

      29	 7.9719544e-01	 1.9289812e-01	 8.3493051e-01	 1.7973387e-01	[ 8.5715887e-01]	 2.0390201e-01


.. parsed-literal::

      30	 8.2262858e-01	 1.8889351e-01	 8.6021158e-01	 1.7528512e-01	[ 8.8441883e-01]	 2.0506787e-01


.. parsed-literal::

      31	 8.4284994e-01	 1.8562493e-01	 8.8092726e-01	 1.7197488e-01	[ 8.9930988e-01]	 2.0939541e-01


.. parsed-literal::

      32	 8.6173027e-01	 1.8567440e-01	 9.0043851e-01	 1.7165454e-01	[ 9.1516832e-01]	 2.0741224e-01


.. parsed-literal::

      33	 8.8207240e-01	 1.8385007e-01	 9.2186798e-01	 1.7051590e-01	[ 9.3019494e-01]	 2.1870971e-01
      34	 8.9807362e-01	 1.8391010e-01	 9.3793981e-01	 1.7093171e-01	[ 9.4644829e-01]	 1.9642472e-01


.. parsed-literal::

      35	 9.1177235e-01	 1.8403395e-01	 9.5173330e-01	 1.7162678e-01	[ 9.5936730e-01]	 2.0876837e-01


.. parsed-literal::

      36	 9.3720137e-01	 1.8481549e-01	 9.7876446e-01	 1.7225114e-01	[ 9.8552180e-01]	 2.1227121e-01


.. parsed-literal::

      37	 9.4704495e-01	 1.8665031e-01	 9.8908612e-01	 1.7273018e-01	[ 9.9928744e-01]	 2.1810675e-01
      38	 9.6169248e-01	 1.8377412e-01	 1.0034466e+00	 1.7027069e-01	[ 1.0165849e+00]	 2.0068860e-01


.. parsed-literal::

      39	 9.7384287e-01	 1.8129513e-01	 1.0157501e+00	 1.6868717e-01	[ 1.0289331e+00]	 2.0660877e-01
      40	 9.8597686e-01	 1.8027834e-01	 1.0284073e+00	 1.6799430e-01	[ 1.0394579e+00]	 1.9937301e-01


.. parsed-literal::

      41	 9.9661592e-01	 1.8030902e-01	 1.0400716e+00	 1.6914581e-01	[ 1.0453284e+00]	 2.0779419e-01
      42	 1.0086421e+00	 1.8082905e-01	 1.0522445e+00	 1.6920437e-01	[ 1.0540014e+00]	 1.7656374e-01


.. parsed-literal::

      43	 1.0137039e+00	 1.8080593e-01	 1.0573059e+00	 1.6932654e-01	[ 1.0570635e+00]	 2.0651865e-01
      44	 1.0281797e+00	 1.8029366e-01	 1.0722748e+00	 1.6871208e-01	[ 1.0722889e+00]	 1.7290974e-01


.. parsed-literal::

      45	 1.0376150e+00	 1.8095626e-01	 1.0824870e+00	 1.6846089e-01	[ 1.0849156e+00]	 2.0719957e-01


.. parsed-literal::

      46	 1.0494624e+00	 1.7905475e-01	 1.0941901e+00	 1.6677770e-01	[ 1.0999773e+00]	 2.0216632e-01


.. parsed-literal::

      47	 1.0558675e+00	 1.7750676e-01	 1.1004838e+00	 1.6551644e-01	[ 1.1061942e+00]	 2.1410418e-01
      48	 1.0643634e+00	 1.7538174e-01	 1.1091163e+00	 1.6403491e-01	[ 1.1129980e+00]	 1.7469978e-01


.. parsed-literal::

      49	 1.0757533e+00	 1.7193939e-01	 1.1210148e+00	 1.6128142e-01	  1.1103701e+00 	 2.0492530e-01


.. parsed-literal::

      50	 1.0828253e+00	 1.6979955e-01	 1.1284919e+00	 1.6076719e-01	  1.1051517e+00 	 2.0388317e-01


.. parsed-literal::

      51	 1.0887767e+00	 1.6966906e-01	 1.1343042e+00	 1.6020178e-01	  1.1125758e+00 	 2.0300531e-01
      52	 1.0929569e+00	 1.6932571e-01	 1.1384692e+00	 1.5989557e-01	[ 1.1163124e+00]	 1.7564130e-01


.. parsed-literal::

      53	 1.0999958e+00	 1.6768254e-01	 1.1457996e+00	 1.5842492e-01	[ 1.1223947e+00]	 2.0264411e-01


.. parsed-literal::

      54	 1.1093616e+00	 1.6676417e-01	 1.1552443e+00	 1.5749978e-01	[ 1.1358231e+00]	 2.0457125e-01


.. parsed-literal::

      55	 1.1195913e+00	 1.6405766e-01	 1.1659503e+00	 1.5553401e-01	[ 1.1416154e+00]	 2.1350336e-01


.. parsed-literal::

      56	 1.1244600e+00	 1.6125883e-01	 1.1715582e+00	 1.5358493e-01	  1.1389141e+00 	 2.1081495e-01


.. parsed-literal::

      57	 1.1313975e+00	 1.6036746e-01	 1.1782895e+00	 1.5304980e-01	[ 1.1476709e+00]	 2.1016550e-01


.. parsed-literal::

      58	 1.1359334e+00	 1.5914001e-01	 1.1829455e+00	 1.5233455e-01	[ 1.1486446e+00]	 2.0822740e-01


.. parsed-literal::

      59	 1.1406544e+00	 1.5816798e-01	 1.1877424e+00	 1.5190478e-01	[ 1.1531039e+00]	 2.0215178e-01
      60	 1.1478975e+00	 1.5683864e-01	 1.1952001e+00	 1.5066693e-01	[ 1.1590858e+00]	 1.8871737e-01


.. parsed-literal::

      61	 1.1551428e+00	 1.5532186e-01	 1.2025379e+00	 1.4950732e-01	[ 1.1705542e+00]	 2.1068454e-01


.. parsed-literal::

      62	 1.1607977e+00	 1.5439759e-01	 1.2083577e+00	 1.4867001e-01	[ 1.1760362e+00]	 2.1834922e-01


.. parsed-literal::

      63	 1.1660245e+00	 1.5334451e-01	 1.2139189e+00	 1.4829942e-01	[ 1.1819644e+00]	 2.0883632e-01


.. parsed-literal::

      64	 1.1710680e+00	 1.5333945e-01	 1.2190594e+00	 1.4841935e-01	  1.1796950e+00 	 2.0810032e-01


.. parsed-literal::

      65	 1.1755257e+00	 1.5272893e-01	 1.2236354e+00	 1.4808217e-01	  1.1793812e+00 	 2.0972800e-01
      66	 1.1836186e+00	 1.5241244e-01	 1.2319247e+00	 1.4793456e-01	[ 1.1831319e+00]	 1.8540406e-01


.. parsed-literal::

      67	 1.1899528e+00	 1.5141361e-01	 1.2384974e+00	 1.4712020e-01	[ 1.1858737e+00]	 2.1060038e-01
      68	 1.1982795e+00	 1.5077343e-01	 1.2469795e+00	 1.4666430e-01	[ 1.1922831e+00]	 1.7411900e-01


.. parsed-literal::

      69	 1.2039631e+00	 1.4978684e-01	 1.2528559e+00	 1.4619851e-01	[ 1.1988641e+00]	 1.9597602e-01
      70	 1.2089083e+00	 1.4954064e-01	 1.2578250e+00	 1.4614629e-01	[ 1.2028502e+00]	 1.9940639e-01


.. parsed-literal::

      71	 1.2128414e+00	 1.4877117e-01	 1.2617553e+00	 1.4592701e-01	[ 1.2035476e+00]	 2.0886540e-01
      72	 1.2179485e+00	 1.4801801e-01	 1.2669790e+00	 1.4592026e-01	  1.2005800e+00 	 1.9890594e-01


.. parsed-literal::

      73	 1.2223258e+00	 1.4776851e-01	 1.2715283e+00	 1.4610298e-01	  1.1993878e+00 	 2.0651197e-01


.. parsed-literal::

      74	 1.2278244e+00	 1.4768425e-01	 1.2771974e+00	 1.4627405e-01	  1.1991743e+00 	 2.1378779e-01


.. parsed-literal::

      75	 1.2324909e+00	 1.4787271e-01	 1.2818708e+00	 1.4635480e-01	[ 1.2036943e+00]	 2.0423341e-01


.. parsed-literal::

      76	 1.2376522e+00	 1.4781146e-01	 1.2870592e+00	 1.4612085e-01	[ 1.2097556e+00]	 2.1024728e-01


.. parsed-literal::

      77	 1.2447715e+00	 1.4729403e-01	 1.2944156e+00	 1.4531969e-01	[ 1.2158425e+00]	 2.0821524e-01


.. parsed-literal::

      78	 1.2516538e+00	 1.4706247e-01	 1.3015521e+00	 1.4512509e-01	[ 1.2167870e+00]	 2.1521497e-01


.. parsed-literal::

      79	 1.2567413e+00	 1.4669917e-01	 1.3066002e+00	 1.4491598e-01	  1.2151051e+00 	 2.0848465e-01


.. parsed-literal::

      80	 1.2624487e+00	 1.4654391e-01	 1.3123826e+00	 1.4495861e-01	  1.2125930e+00 	 2.0701218e-01


.. parsed-literal::

      81	 1.2642630e+00	 1.4605366e-01	 1.3146589e+00	 1.4471371e-01	  1.1925686e+00 	 2.1285915e-01


.. parsed-literal::

      82	 1.2700147e+00	 1.4608403e-01	 1.3202444e+00	 1.4475171e-01	  1.2024745e+00 	 2.1906066e-01


.. parsed-literal::

      83	 1.2725297e+00	 1.4600189e-01	 1.3227155e+00	 1.4469592e-01	  1.2060084e+00 	 2.1829510e-01
      84	 1.2764697e+00	 1.4562292e-01	 1.3268062e+00	 1.4453204e-01	  1.2044713e+00 	 1.8690825e-01


.. parsed-literal::

      85	 1.2811080e+00	 1.4529771e-01	 1.3315997e+00	 1.4434617e-01	  1.2012290e+00 	 2.0461011e-01
      86	 1.2867837e+00	 1.4439408e-01	 1.3375028e+00	 1.4420140e-01	  1.1943792e+00 	 1.8989229e-01


.. parsed-literal::

      87	 1.2915629e+00	 1.4372126e-01	 1.3423386e+00	 1.4399085e-01	  1.1926668e+00 	 2.1368909e-01


.. parsed-literal::

      88	 1.2972974e+00	 1.4313566e-01	 1.3483125e+00	 1.4413476e-01	  1.1854790e+00 	 2.1576953e-01


.. parsed-literal::

      89	 1.3011298e+00	 1.4259413e-01	 1.3522749e+00	 1.4403920e-01	  1.1779765e+00 	 2.1175671e-01


.. parsed-literal::

      90	 1.3053473e+00	 1.4271665e-01	 1.3564166e+00	 1.4428908e-01	  1.1828652e+00 	 2.2578597e-01


.. parsed-literal::

      91	 1.3089565e+00	 1.4273587e-01	 1.3600340e+00	 1.4447893e-01	  1.1867228e+00 	 2.2115850e-01
      92	 1.3118026e+00	 1.4252139e-01	 1.3628733e+00	 1.4435858e-01	  1.1906085e+00 	 2.0117879e-01


.. parsed-literal::

      93	 1.3179573e+00	 1.4187178e-01	 1.3692221e+00	 1.4381681e-01	  1.1897034e+00 	 2.0851231e-01


.. parsed-literal::

      94	 1.3213670e+00	 1.4125440e-01	 1.3727614e+00	 1.4340869e-01	  1.1925654e+00 	 3.1844711e-01
      95	 1.3258502e+00	 1.4079058e-01	 1.3773716e+00	 1.4286498e-01	  1.1907108e+00 	 2.0300221e-01


.. parsed-literal::

      96	 1.3304617e+00	 1.4036843e-01	 1.3821781e+00	 1.4230530e-01	  1.1813155e+00 	 2.1937966e-01


.. parsed-literal::

      97	 1.3340088e+00	 1.4021686e-01	 1.3859606e+00	 1.4203672e-01	  1.1765177e+00 	 2.1662164e-01


.. parsed-literal::

      98	 1.3379829e+00	 1.4003608e-01	 1.3900271e+00	 1.4190959e-01	  1.1704039e+00 	 2.1257544e-01
      99	 1.3425702e+00	 1.4010225e-01	 1.3946987e+00	 1.4223581e-01	  1.1637956e+00 	 1.7706966e-01


.. parsed-literal::

     100	 1.3462947e+00	 1.3996558e-01	 1.3984601e+00	 1.4229150e-01	  1.1676995e+00 	 1.9655514e-01


.. parsed-literal::

     101	 1.3505263e+00	 1.3939575e-01	 1.4027496e+00	 1.4203357e-01	  1.1686678e+00 	 2.0682526e-01
     102	 1.3554898e+00	 1.3828404e-01	 1.4077888e+00	 1.4127903e-01	  1.1725090e+00 	 1.7955661e-01


.. parsed-literal::

     103	 1.3595285e+00	 1.3697948e-01	 1.4118699e+00	 1.4020235e-01	  1.1806183e+00 	 2.1432829e-01
     104	 1.3638077e+00	 1.3627292e-01	 1.4161632e+00	 1.3958588e-01	  1.1877651e+00 	 1.8587995e-01


.. parsed-literal::

     105	 1.3675973e+00	 1.3559730e-01	 1.4199584e+00	 1.3891010e-01	  1.1900357e+00 	 1.9725704e-01


.. parsed-literal::

     106	 1.3707066e+00	 1.3557400e-01	 1.4231082e+00	 1.3859985e-01	  1.1953296e+00 	 2.1754026e-01
     107	 1.3739322e+00	 1.3522205e-01	 1.4264020e+00	 1.3829543e-01	  1.1935180e+00 	 2.0530868e-01


.. parsed-literal::

     108	 1.3786775e+00	 1.3509547e-01	 1.4312760e+00	 1.3808096e-01	  1.1934069e+00 	 1.7238188e-01


.. parsed-literal::

     109	 1.3821097e+00	 1.3483542e-01	 1.4348637e+00	 1.3761526e-01	  1.1948472e+00 	 2.0375347e-01


.. parsed-literal::

     110	 1.3856741e+00	 1.3481501e-01	 1.4384795e+00	 1.3742547e-01	  1.2007311e+00 	 2.2072506e-01


.. parsed-literal::

     111	 1.3889416e+00	 1.3450947e-01	 1.4418315e+00	 1.3693056e-01	  1.2029570e+00 	 2.0287824e-01


.. parsed-literal::

     112	 1.3918091e+00	 1.3450351e-01	 1.4448246e+00	 1.3666248e-01	  1.2069122e+00 	 2.1031666e-01


.. parsed-literal::

     113	 1.3948669e+00	 1.3410933e-01	 1.4479845e+00	 1.3606236e-01	  1.2036541e+00 	 2.0617890e-01


.. parsed-literal::

     114	 1.3989809e+00	 1.3334982e-01	 1.4523848e+00	 1.3518641e-01	  1.1921862e+00 	 2.1957827e-01
     115	 1.4017354e+00	 1.3307100e-01	 1.4553202e+00	 1.3474802e-01	  1.1875793e+00 	 2.0536399e-01


.. parsed-literal::

     116	 1.4044963e+00	 1.3278277e-01	 1.4580502e+00	 1.3454965e-01	  1.1896178e+00 	 2.1388936e-01


.. parsed-literal::

     117	 1.4092479e+00	 1.3216725e-01	 1.4628528e+00	 1.3421698e-01	  1.1931823e+00 	 2.1669126e-01


.. parsed-literal::

     118	 1.4122754e+00	 1.3167575e-01	 1.4658317e+00	 1.3368308e-01	  1.2034905e+00 	 2.1260834e-01
     119	 1.4156328e+00	 1.3130916e-01	 1.4691269e+00	 1.3337768e-01	  1.2110602e+00 	 1.8978071e-01


.. parsed-literal::

     120	 1.4189931e+00	 1.3101280e-01	 1.4724878e+00	 1.3303548e-01	[ 1.2182677e+00]	 2.1422505e-01


.. parsed-literal::

     121	 1.4213453e+00	 1.3083306e-01	 1.4749558e+00	 1.3244441e-01	[ 1.2218491e+00]	 2.1438217e-01


.. parsed-literal::

     122	 1.4238231e+00	 1.3075213e-01	 1.4773968e+00	 1.3235619e-01	  1.2180545e+00 	 2.1877027e-01
     123	 1.4261659e+00	 1.3075575e-01	 1.4797732e+00	 1.3223157e-01	  1.2147632e+00 	 1.9212985e-01


.. parsed-literal::

     124	 1.4289578e+00	 1.3069242e-01	 1.4825352e+00	 1.3190823e-01	  1.2150466e+00 	 2.1730661e-01
     125	 1.4306402e+00	 1.3066423e-01	 1.4842225e+00	 1.3133844e-01	  1.2212373e+00 	 1.9789195e-01


.. parsed-literal::

     126	 1.4348128e+00	 1.3050805e-01	 1.4882388e+00	 1.3109058e-01	[ 1.2261339e+00]	 1.8746209e-01


.. parsed-literal::

     127	 1.4366126e+00	 1.3037410e-01	 1.4899767e+00	 1.3092440e-01	[ 1.2304405e+00]	 2.1791625e-01
     128	 1.4389234e+00	 1.3017496e-01	 1.4922821e+00	 1.3063677e-01	[ 1.2347893e+00]	 2.0486140e-01


.. parsed-literal::

     129	 1.4424711e+00	 1.2994280e-01	 1.4959481e+00	 1.3034854e-01	[ 1.2358799e+00]	 2.1672225e-01


.. parsed-literal::

     130	 1.4445790e+00	 1.2972759e-01	 1.4981757e+00	 1.3024413e-01	  1.2325310e+00 	 3.2845354e-01


.. parsed-literal::

     131	 1.4475600e+00	 1.2964547e-01	 1.5013037e+00	 1.3010344e-01	  1.2277842e+00 	 2.1049690e-01
     132	 1.4498768e+00	 1.2962262e-01	 1.5036826e+00	 1.3003694e-01	  1.2252176e+00 	 1.9960570e-01


.. parsed-literal::

     133	 1.4528483e+00	 1.2964931e-01	 1.5067746e+00	 1.2984375e-01	  1.2159645e+00 	 2.1896696e-01


.. parsed-literal::

     134	 1.4552667e+00	 1.2976297e-01	 1.5092346e+00	 1.2975718e-01	  1.2185731e+00 	 2.1316123e-01


.. parsed-literal::

     135	 1.4572603e+00	 1.2979172e-01	 1.5111635e+00	 1.2967663e-01	  1.2178334e+00 	 2.0356607e-01


.. parsed-literal::

     136	 1.4599782e+00	 1.2989019e-01	 1.5139103e+00	 1.2951331e-01	  1.2176213e+00 	 2.0165324e-01
     137	 1.4616196e+00	 1.2982095e-01	 1.5155991e+00	 1.2945717e-01	  1.2097233e+00 	 1.9706011e-01


.. parsed-literal::

     138	 1.4631316e+00	 1.2970387e-01	 1.5171626e+00	 1.2936559e-01	  1.2055089e+00 	 2.1127915e-01


.. parsed-literal::

     139	 1.4660623e+00	 1.2947895e-01	 1.5203339e+00	 1.2919812e-01	  1.1902175e+00 	 2.1215630e-01


.. parsed-literal::

     140	 1.4668307e+00	 1.2931221e-01	 1.5212296e+00	 1.2916371e-01	  1.1769045e+00 	 2.0847416e-01


.. parsed-literal::

     141	 1.4691829e+00	 1.2924607e-01	 1.5235409e+00	 1.2905871e-01	  1.1789278e+00 	 2.0549536e-01


.. parsed-literal::

     142	 1.4709820e+00	 1.2918182e-01	 1.5253397e+00	 1.2894624e-01	  1.1770853e+00 	 2.0529652e-01


.. parsed-literal::

     143	 1.4726830e+00	 1.2909906e-01	 1.5270534e+00	 1.2883453e-01	  1.1725694e+00 	 2.0741916e-01


.. parsed-literal::

     144	 1.4753514e+00	 1.2908649e-01	 1.5297614e+00	 1.2876401e-01	  1.1670219e+00 	 2.1286583e-01


.. parsed-literal::

     145	 1.4768161e+00	 1.2901967e-01	 1.5313118e+00	 1.2862059e-01	  1.1550838e+00 	 3.1969428e-01


.. parsed-literal::

     146	 1.4789490e+00	 1.2907533e-01	 1.5335020e+00	 1.2864281e-01	  1.1522399e+00 	 2.0673013e-01


.. parsed-literal::

     147	 1.4806685e+00	 1.2909653e-01	 1.5352429e+00	 1.2866785e-01	  1.1550295e+00 	 2.1680522e-01


.. parsed-literal::

     148	 1.4829023e+00	 1.2903721e-01	 1.5375463e+00	 1.2867759e-01	  1.1578464e+00 	 2.0958900e-01


.. parsed-literal::

     149	 1.4847247e+00	 1.2909025e-01	 1.5394598e+00	 1.2883421e-01	  1.1586378e+00 	 2.0260334e-01


.. parsed-literal::

     150	 1.4868458e+00	 1.2903583e-01	 1.5415268e+00	 1.2890854e-01	  1.1672748e+00 	 2.2122598e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.03 s, total: 2min 7s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fbcf4929990>



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


.. parsed-literal::

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
    CPU times: user 2.03 s, sys: 42 ms, total: 2.07 s
    Wall time: 617 ms


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

