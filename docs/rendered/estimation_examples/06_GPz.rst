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
       1	-3.2989732e-01	 3.1685173e-01	-3.2015231e-01	 3.3448597e-01	[-3.5558363e-01]	 4.6462631e-01


.. parsed-literal::

       2	-2.5959599e-01	 3.0563133e-01	-2.3462790e-01	 3.2378071e-01	[-2.9295619e-01]	 2.2861981e-01


.. parsed-literal::

       3	-2.1677705e-01	 2.8589820e-01	-1.7398390e-01	 3.0425888e-01	[-2.5600977e-01]	 2.9055953e-01
       4	-1.8471719e-01	 2.6193682e-01	-1.4256594e-01	 2.8159401e-01	 -2.6750033e-01 	 1.7184186e-01


.. parsed-literal::

       5	-9.3370436e-02	 2.5355349e-01	-5.6633395e-02	 2.7153312e-01	[-1.4853293e-01]	 2.0075893e-01


.. parsed-literal::

       6	-5.9800279e-02	 2.4881388e-01	-2.8153482e-02	 2.6627197e-01	[-9.7620410e-02]	 2.0639038e-01


.. parsed-literal::

       7	-4.1312651e-02	 2.4588330e-01	-1.6317710e-02	 2.6280401e-01	[-8.7172815e-02]	 2.1317458e-01
       8	-2.9736877e-02	 2.4405426e-01	-8.6633692e-03	 2.6033251e-01	[-7.7887599e-02]	 1.8716097e-01


.. parsed-literal::

       9	-1.6844128e-02	 2.4179400e-01	 1.1031667e-03	 2.5718784e-01	[-6.5033632e-02]	 1.9751358e-01


.. parsed-literal::

      10	-2.9139857e-03	 2.3906498e-01	 1.2913813e-02	 2.5420115e-01	[-5.4276438e-02]	 2.1048641e-01


.. parsed-literal::

      11	-1.8719659e-03	 2.3900384e-01	 1.2634656e-02	 2.5490660e-01	[-5.1118031e-02]	 2.1348023e-01


.. parsed-literal::

      12	 5.3566221e-03	 2.3783698e-01	 1.9366599e-02	 2.5400549e-01	 -5.2889952e-02 	 2.1327424e-01
      13	 8.0735481e-03	 2.3729199e-01	 2.1860339e-02	 2.5360162e-01	[-4.9336552e-02]	 1.7732382e-01


.. parsed-literal::

      14	 1.1266729e-02	 2.3665114e-01	 2.5113662e-02	 2.5317089e-01	[-4.8009895e-02]	 2.0806408e-01


.. parsed-literal::

      15	 2.2183726e-02	 2.3470844e-01	 3.6772693e-02	 2.5148291e-01	[-3.7287185e-02]	 2.0882940e-01


.. parsed-literal::

      16	 1.2107091e-01	 2.2347316e-01	 1.4164554e-01	 2.4063218e-01	[ 8.9707046e-02]	 2.0909882e-01
      17	 1.7239432e-01	 2.1882880e-01	 1.9478997e-01	 2.3613914e-01	[ 1.5588589e-01]	 1.9217610e-01


.. parsed-literal::

      18	 2.8982127e-01	 2.1702584e-01	 3.1546923e-01	 2.3314956e-01	[ 2.7351213e-01]	 2.1667099e-01
      19	 3.4406621e-01	 2.1539098e-01	 3.7645533e-01	 2.3142145e-01	[ 2.9954447e-01]	 1.7522311e-01


.. parsed-literal::

      20	 3.9273172e-01	 2.1351433e-01	 4.2472043e-01	 2.3030429e-01	[ 3.6422476e-01]	 1.8796134e-01


.. parsed-literal::

      21	 4.3438538e-01	 2.0958148e-01	 4.6594661e-01	 2.2738141e-01	[ 4.1716192e-01]	 2.0809507e-01
      22	 4.9527437e-01	 2.0662274e-01	 5.2806890e-01	 2.2571423e-01	[ 4.8868725e-01]	 1.8887281e-01


.. parsed-literal::

      23	 5.6190256e-01	 2.0764015e-01	 5.9813333e-01	 2.2581091e-01	[ 5.8009254e-01]	 1.9006562e-01


.. parsed-literal::

      24	 5.9336990e-01	 2.0634283e-01	 6.3198548e-01	 2.2400075e-01	[ 6.1271750e-01]	 2.1099591e-01
      25	 6.2336923e-01	 2.0371187e-01	 6.6092717e-01	 2.2159109e-01	[ 6.4494691e-01]	 1.9868255e-01


.. parsed-literal::

      26	 6.4842190e-01	 2.0187226e-01	 6.8478093e-01	 2.1976780e-01	[ 6.7500800e-01]	 2.0539618e-01
      27	 6.9200099e-01	 1.9837246e-01	 7.2707414e-01	 2.1506123e-01	[ 7.2887435e-01]	 2.0087504e-01


.. parsed-literal::

      28	 7.1075873e-01	 1.9745031e-01	 7.4568825e-01	 2.1278180e-01	[ 7.4852926e-01]	 2.0151424e-01
      29	 7.5050431e-01	 1.9606317e-01	 7.8846548e-01	 2.1265051e-01	[ 8.0440779e-01]	 1.9655919e-01


.. parsed-literal::

      30	 7.7586432e-01	 1.9429048e-01	 8.1421228e-01	 2.1000610e-01	[ 8.3371206e-01]	 2.1025491e-01
      31	 7.9249145e-01	 1.9308105e-01	 8.3075875e-01	 2.0977287e-01	[ 8.5371656e-01]	 2.0048761e-01


.. parsed-literal::

      32	 8.2298031e-01	 1.9219838e-01	 8.6242991e-01	 2.1162498e-01	[ 8.8503887e-01]	 2.0898438e-01
      33	 8.5763588e-01	 1.9661754e-01	 8.9766469e-01	 2.1968231e-01	[ 9.1625765e-01]	 1.9562674e-01


.. parsed-literal::

      34	 8.6929404e-01	 1.9719956e-01	 9.1429435e-01	 2.1894450e-01	  9.1444112e-01 	 2.0894814e-01


.. parsed-literal::

      35	 9.0605321e-01	 1.9154830e-01	 9.4864593e-01	 2.1306171e-01	[ 9.5566236e-01]	 2.1953535e-01


.. parsed-literal::

      36	 9.2162577e-01	 1.8945062e-01	 9.6387065e-01	 2.1096679e-01	[ 9.7161745e-01]	 2.0377398e-01
      37	 9.3493546e-01	 1.8664519e-01	 9.7727576e-01	 2.0821631e-01	[ 9.8359388e-01]	 1.7508030e-01


.. parsed-literal::

      38	 9.5607446e-01	 1.8267997e-01	 9.9918213e-01	 2.0346844e-01	[ 1.0038869e+00]	 2.1232176e-01


.. parsed-literal::

      39	 9.7018390e-01	 1.8080058e-01	 1.0138841e+00	 2.0124138e-01	[ 1.0209624e+00]	 2.1099782e-01


.. parsed-literal::

      40	 9.8613297e-01	 1.7810210e-01	 1.0306705e+00	 1.9723897e-01	[ 1.0339739e+00]	 2.1923971e-01
      41	 9.9808761e-01	 1.7564405e-01	 1.0428606e+00	 1.9424525e-01	[ 1.0519443e+00]	 1.9414759e-01


.. parsed-literal::

      42	 1.0105521e+00	 1.7442716e-01	 1.0556855e+00	 1.9282022e-01	[ 1.0671898e+00]	 1.9916081e-01


.. parsed-literal::

      43	 1.0258702e+00	 1.7248460e-01	 1.0713430e+00	 1.9085259e-01	[ 1.0842611e+00]	 2.0764613e-01
      44	 1.0400870e+00	 1.7165967e-01	 1.0856437e+00	 1.9010066e-01	[ 1.0954572e+00]	 1.8525839e-01


.. parsed-literal::

      45	 1.0500853e+00	 1.7025210e-01	 1.0958615e+00	 1.8854263e-01	[ 1.1030903e+00]	 2.2801828e-01
      46	 1.0592556e+00	 1.6959182e-01	 1.1050069e+00	 1.8767923e-01	[ 1.1089601e+00]	 1.6705894e-01


.. parsed-literal::

      47	 1.0716853e+00	 1.6812750e-01	 1.1177781e+00	 1.8626965e-01	[ 1.1164405e+00]	 1.8852735e-01
      48	 1.0778947e+00	 1.6706535e-01	 1.1242441e+00	 1.8456277e-01	[ 1.1215410e+00]	 1.7212200e-01


.. parsed-literal::

      49	 1.0840730e+00	 1.6631683e-01	 1.1301790e+00	 1.8384783e-01	[ 1.1275885e+00]	 2.0812345e-01


.. parsed-literal::

      50	 1.0918975e+00	 1.6500142e-01	 1.1381314e+00	 1.8252783e-01	[ 1.1335378e+00]	 2.2270679e-01


.. parsed-literal::

      51	 1.0974537e+00	 1.6371721e-01	 1.1437955e+00	 1.8082784e-01	[ 1.1388648e+00]	 2.1428704e-01


.. parsed-literal::

      52	 1.1052690e+00	 1.6175228e-01	 1.1520974e+00	 1.7908514e-01	[ 1.1414195e+00]	 2.1279740e-01
      53	 1.1134748e+00	 1.6014172e-01	 1.1605396e+00	 1.7700373e-01	[ 1.1482354e+00]	 1.9129682e-01


.. parsed-literal::

      54	 1.1213810e+00	 1.5906095e-01	 1.1686825e+00	 1.7597812e-01	[ 1.1528988e+00]	 1.8909836e-01


.. parsed-literal::

      55	 1.1328432e+00	 1.5751283e-01	 1.1805641e+00	 1.7485963e-01	[ 1.1612373e+00]	 2.1643662e-01


.. parsed-literal::

      56	 1.1414775e+00	 1.5567267e-01	 1.1901342e+00	 1.7435466e-01	  1.1555266e+00 	 2.1138501e-01
      57	 1.1521281e+00	 1.5542733e-01	 1.2004772e+00	 1.7429388e-01	[ 1.1689430e+00]	 1.9358659e-01


.. parsed-literal::

      58	 1.1589034e+00	 1.5386085e-01	 1.2073648e+00	 1.7286846e-01	[ 1.1780701e+00]	 2.1035314e-01
      59	 1.1682461e+00	 1.5223291e-01	 1.2169596e+00	 1.7207626e-01	[ 1.1860007e+00]	 1.8468022e-01


.. parsed-literal::

      60	 1.1766328e+00	 1.4947752e-01	 1.2255993e+00	 1.7004301e-01	[ 1.1950518e+00]	 2.1314645e-01


.. parsed-literal::

      61	 1.1843201e+00	 1.4913216e-01	 1.2332108e+00	 1.6976464e-01	[ 1.1995517e+00]	 2.0542598e-01
      62	 1.1889970e+00	 1.4820736e-01	 1.2381084e+00	 1.6914968e-01	  1.1989172e+00 	 1.7740703e-01


.. parsed-literal::

      63	 1.1950608e+00	 1.4743806e-01	 1.2442766e+00	 1.6815353e-01	[ 1.2044772e+00]	 2.0624590e-01


.. parsed-literal::

      64	 1.2032511e+00	 1.4569385e-01	 1.2526286e+00	 1.6660961e-01	[ 1.2084137e+00]	 2.1912217e-01
      65	 1.2156727e+00	 1.4350373e-01	 1.2657044e+00	 1.6413719e-01	[ 1.2132993e+00]	 1.7424798e-01


.. parsed-literal::

      66	 1.2226145e+00	 1.4250132e-01	 1.2727939e+00	 1.6389692e-01	[ 1.2182810e+00]	 2.0632124e-01


.. parsed-literal::

      67	 1.2299560e+00	 1.4211461e-01	 1.2799971e+00	 1.6289290e-01	[ 1.2277259e+00]	 2.0912242e-01


.. parsed-literal::

      68	 1.2335576e+00	 1.4203902e-01	 1.2835779e+00	 1.6277639e-01	[ 1.2307833e+00]	 2.1442676e-01


.. parsed-literal::

      69	 1.2388583e+00	 1.4182538e-01	 1.2891349e+00	 1.6273216e-01	[ 1.2315201e+00]	 2.0706964e-01


.. parsed-literal::

      70	 1.2437649e+00	 1.4141254e-01	 1.2945548e+00	 1.6289677e-01	  1.2253758e+00 	 2.0820284e-01
      71	 1.2503970e+00	 1.4076993e-01	 1.3011592e+00	 1.6219255e-01	  1.2286914e+00 	 1.7701650e-01


.. parsed-literal::

      72	 1.2549348e+00	 1.4009882e-01	 1.3056845e+00	 1.6149234e-01	  1.2305640e+00 	 2.0962429e-01


.. parsed-literal::

      73	 1.2599798e+00	 1.3942056e-01	 1.3107811e+00	 1.6074859e-01	[ 1.2322814e+00]	 2.0360756e-01
      74	 1.2692946e+00	 1.3844009e-01	 1.3200910e+00	 1.5955610e-01	[ 1.2384515e+00]	 1.6654634e-01


.. parsed-literal::

      75	 1.2743894e+00	 1.3792841e-01	 1.3252874e+00	 1.5925102e-01	[ 1.2400038e+00]	 2.9100394e-01
      76	 1.2791397e+00	 1.3775348e-01	 1.3299636e+00	 1.5895211e-01	[ 1.2457993e+00]	 1.7473841e-01


.. parsed-literal::

      77	 1.2839111e+00	 1.3747944e-01	 1.3347508e+00	 1.5857719e-01	[ 1.2526045e+00]	 2.1249866e-01


.. parsed-literal::

      78	 1.2896943e+00	 1.3715628e-01	 1.3406363e+00	 1.5805087e-01	[ 1.2560823e+00]	 2.0164394e-01
      79	 1.2960669e+00	 1.3604312e-01	 1.3472378e+00	 1.5653390e-01	[ 1.2620310e+00]	 1.7422533e-01


.. parsed-literal::

      80	 1.3016355e+00	 1.3557764e-01	 1.3528030e+00	 1.5616334e-01	[ 1.2642917e+00]	 2.0497513e-01


.. parsed-literal::

      81	 1.3070893e+00	 1.3521627e-01	 1.3582378e+00	 1.5593968e-01	  1.2628819e+00 	 2.1021056e-01


.. parsed-literal::

      82	 1.3128724e+00	 1.3454314e-01	 1.3643858e+00	 1.5516831e-01	  1.2622232e+00 	 2.1141839e-01
      83	 1.3205839e+00	 1.3395268e-01	 1.3724893e+00	 1.5414592e-01	  1.2600389e+00 	 1.8882489e-01


.. parsed-literal::

      84	 1.3257312e+00	 1.3378891e-01	 1.3778119e+00	 1.5392349e-01	  1.2621006e+00 	 1.9985318e-01
      85	 1.3320265e+00	 1.3401365e-01	 1.3842188e+00	 1.5480684e-01	[ 1.2665973e+00]	 1.7703247e-01


.. parsed-literal::

      86	 1.3366851e+00	 1.3426350e-01	 1.3890837e+00	 1.5534801e-01	  1.2658323e+00 	 2.0877838e-01


.. parsed-literal::

      87	 1.3410961e+00	 1.3451230e-01	 1.3935255e+00	 1.5611225e-01	  1.2656790e+00 	 2.0861697e-01


.. parsed-literal::

      88	 1.3435435e+00	 1.3461387e-01	 1.3960941e+00	 1.5641396e-01	  1.2644933e+00 	 2.0676899e-01


.. parsed-literal::

      89	 1.3463972e+00	 1.3480029e-01	 1.3989130e+00	 1.5671369e-01	  1.2651335e+00 	 2.2157025e-01


.. parsed-literal::

      90	 1.3512276e+00	 1.3467880e-01	 1.4039473e+00	 1.5605516e-01	[ 1.2691395e+00]	 2.0509601e-01


.. parsed-literal::

      91	 1.3531544e+00	 1.3507365e-01	 1.4061097e+00	 1.5642150e-01	  1.2650706e+00 	 2.0793009e-01


.. parsed-literal::

      92	 1.3582594e+00	 1.3449004e-01	 1.4110523e+00	 1.5550927e-01	[ 1.2741261e+00]	 2.0708013e-01


.. parsed-literal::

      93	 1.3600272e+00	 1.3418957e-01	 1.4128082e+00	 1.5513032e-01	[ 1.2772392e+00]	 2.1726084e-01


.. parsed-literal::

      94	 1.3635635e+00	 1.3373843e-01	 1.4163450e+00	 1.5463331e-01	[ 1.2797707e+00]	 2.0201254e-01


.. parsed-literal::

      95	 1.3682293e+00	 1.3332986e-01	 1.4209818e+00	 1.5418901e-01	[ 1.2836700e+00]	 2.0513129e-01


.. parsed-literal::

      96	 1.3704682e+00	 1.3301550e-01	 1.4235917e+00	 1.5371708e-01	  1.2745291e+00 	 2.0345545e-01


.. parsed-literal::

      97	 1.3766457e+00	 1.3277110e-01	 1.4295792e+00	 1.5339942e-01	[ 1.2854599e+00]	 2.1788549e-01
      98	 1.3794215e+00	 1.3272994e-01	 1.4323583e+00	 1.5319554e-01	[ 1.2893130e+00]	 1.8257308e-01


.. parsed-literal::

      99	 1.3831536e+00	 1.3248918e-01	 1.4363192e+00	 1.5269281e-01	[ 1.2904700e+00]	 1.9307876e-01


.. parsed-literal::

     100	 1.3855606e+00	 1.3254906e-01	 1.4389546e+00	 1.5217772e-01	[ 1.2913048e+00]	 2.5604153e-01


.. parsed-literal::

     101	 1.3889463e+00	 1.3203361e-01	 1.4423969e+00	 1.5165599e-01	[ 1.2937490e+00]	 2.0514774e-01


.. parsed-literal::

     102	 1.3914569e+00	 1.3171086e-01	 1.4449750e+00	 1.5139380e-01	[ 1.2939073e+00]	 2.1241689e-01


.. parsed-literal::

     103	 1.3931418e+00	 1.3145667e-01	 1.4467293e+00	 1.5116673e-01	  1.2935561e+00 	 2.1350002e-01


.. parsed-literal::

     104	 1.3955771e+00	 1.3112300e-01	 1.4491579e+00	 1.5080257e-01	  1.2936627e+00 	 2.0860338e-01


.. parsed-literal::

     105	 1.3978889e+00	 1.3084222e-01	 1.4514341e+00	 1.5066593e-01	[ 1.2950271e+00]	 2.2329688e-01
     106	 1.4014117e+00	 1.3032743e-01	 1.4549332e+00	 1.5042345e-01	[ 1.2971345e+00]	 1.7987919e-01


.. parsed-literal::

     107	 1.4043788e+00	 1.2993936e-01	 1.4579387e+00	 1.5022448e-01	[ 1.2984219e+00]	 2.0838690e-01


.. parsed-literal::

     108	 1.4099956e+00	 1.2909240e-01	 1.4637633e+00	 1.5000819e-01	  1.2954354e+00 	 2.1469402e-01


.. parsed-literal::

     109	 1.4126960e+00	 1.2876208e-01	 1.4667116e+00	 1.4972689e-01	  1.2973995e+00 	 2.1306443e-01


.. parsed-literal::

     110	 1.4159740e+00	 1.2874933e-01	 1.4697899e+00	 1.4953087e-01	[ 1.3014038e+00]	 2.2171283e-01


.. parsed-literal::

     111	 1.4179457e+00	 1.2863696e-01	 1.4717582e+00	 1.4937727e-01	[ 1.3016858e+00]	 2.1961498e-01
     112	 1.4207470e+00	 1.2846919e-01	 1.4745876e+00	 1.4907251e-01	  1.2994582e+00 	 1.9694400e-01


.. parsed-literal::

     113	 1.4223656e+00	 1.2808250e-01	 1.4764152e+00	 1.4874700e-01	  1.2978376e+00 	 2.1201730e-01


.. parsed-literal::

     114	 1.4257666e+00	 1.2797911e-01	 1.4797338e+00	 1.4858103e-01	  1.2969738e+00 	 2.1224070e-01


.. parsed-literal::

     115	 1.4274101e+00	 1.2787692e-01	 1.4813731e+00	 1.4850916e-01	  1.2960981e+00 	 2.2380805e-01
     116	 1.4290499e+00	 1.2775280e-01	 1.4830611e+00	 1.4842373e-01	  1.2958694e+00 	 1.9644809e-01


.. parsed-literal::

     117	 1.4305271e+00	 1.2750966e-01	 1.4846526e+00	 1.4840417e-01	  1.2932481e+00 	 1.8945456e-01


.. parsed-literal::

     118	 1.4322411e+00	 1.2742861e-01	 1.4863603e+00	 1.4829429e-01	  1.2963953e+00 	 2.1911979e-01


.. parsed-literal::

     119	 1.4347074e+00	 1.2724318e-01	 1.4889125e+00	 1.4802992e-01	  1.3000101e+00 	 2.0979095e-01


.. parsed-literal::

     120	 1.4359551e+00	 1.2714597e-01	 1.4902084e+00	 1.4781284e-01	  1.2996779e+00 	 2.1048689e-01


.. parsed-literal::

     121	 1.4399174e+00	 1.2691619e-01	 1.4943271e+00	 1.4752075e-01	  1.2957635e+00 	 2.0950723e-01


.. parsed-literal::

     122	 1.4407583e+00	 1.2662446e-01	 1.4954032e+00	 1.4643972e-01	  1.2810124e+00 	 2.1464133e-01


.. parsed-literal::

     123	 1.4442298e+00	 1.2666890e-01	 1.4986882e+00	 1.4696909e-01	  1.2889522e+00 	 2.0918489e-01


.. parsed-literal::

     124	 1.4456043e+00	 1.2667656e-01	 1.5000232e+00	 1.4720391e-01	  1.2887707e+00 	 2.0959353e-01


.. parsed-literal::

     125	 1.4478549e+00	 1.2663860e-01	 1.5023051e+00	 1.4743473e-01	  1.2870210e+00 	 2.0771861e-01


.. parsed-literal::

     126	 1.4501456e+00	 1.2658453e-01	 1.5046816e+00	 1.4749024e-01	  1.2860624e+00 	 2.0527792e-01


.. parsed-literal::

     127	 1.4526099e+00	 1.2646639e-01	 1.5072137e+00	 1.4724742e-01	  1.2882755e+00 	 2.0868230e-01


.. parsed-literal::

     128	 1.4552988e+00	 1.2620571e-01	 1.5100348e+00	 1.4666955e-01	  1.2939684e+00 	 2.0782709e-01


.. parsed-literal::

     129	 1.4561994e+00	 1.2622465e-01	 1.5109361e+00	 1.4642804e-01	  1.2963213e+00 	 3.3238220e-01


.. parsed-literal::

     130	 1.4574520e+00	 1.2608437e-01	 1.5121898e+00	 1.4615375e-01	  1.2990043e+00 	 2.2258830e-01


.. parsed-literal::

     131	 1.4596452e+00	 1.2583054e-01	 1.5143731e+00	 1.4574368e-01	[ 1.3035720e+00]	 2.0469570e-01


.. parsed-literal::

     132	 1.4613250e+00	 1.2561903e-01	 1.5160786e+00	 1.4547250e-01	  1.3021792e+00 	 2.1414590e-01
     133	 1.4631988e+00	 1.2540040e-01	 1.5179336e+00	 1.4533268e-01	[ 1.3051993e+00]	 1.9897962e-01


.. parsed-literal::

     134	 1.4657228e+00	 1.2513478e-01	 1.5204809e+00	 1.4509990e-01	[ 1.3059596e+00]	 2.0943999e-01


.. parsed-literal::

     135	 1.4676808e+00	 1.2483577e-01	 1.5225937e+00	 1.4510978e-01	[ 1.3082816e+00]	 2.0891428e-01


.. parsed-literal::

     136	 1.4702573e+00	 1.2468067e-01	 1.5251654e+00	 1.4474039e-01	[ 1.3114932e+00]	 2.1554708e-01


.. parsed-literal::

     137	 1.4719379e+00	 1.2459176e-01	 1.5268739e+00	 1.4449264e-01	[ 1.3121869e+00]	 2.0562053e-01
     138	 1.4745870e+00	 1.2445615e-01	 1.5296196e+00	 1.4415131e-01	  1.3108274e+00 	 1.8482256e-01


.. parsed-literal::

     139	 1.4757890e+00	 1.2437202e-01	 1.5309127e+00	 1.4395759e-01	  1.3108349e+00 	 3.1833363e-01
     140	 1.4777110e+00	 1.2434369e-01	 1.5328879e+00	 1.4389797e-01	  1.3089164e+00 	 1.8678641e-01


.. parsed-literal::

     141	 1.4798380e+00	 1.2429899e-01	 1.5350689e+00	 1.4386720e-01	  1.3073134e+00 	 2.0669413e-01


.. parsed-literal::

     142	 1.4824052e+00	 1.2422690e-01	 1.5377166e+00	 1.4372177e-01	  1.3073886e+00 	 2.1501422e-01


.. parsed-literal::

     143	 1.4833691e+00	 1.2414895e-01	 1.5388968e+00	 1.4380863e-01	  1.3062994e+00 	 2.1998405e-01


.. parsed-literal::

     144	 1.4861465e+00	 1.2416055e-01	 1.5415477e+00	 1.4358701e-01	  1.3118606e+00 	 2.0997739e-01


.. parsed-literal::

     145	 1.4874684e+00	 1.2411799e-01	 1.5428444e+00	 1.4341251e-01	[ 1.3152980e+00]	 2.1404409e-01


.. parsed-literal::

     146	 1.4895547e+00	 1.2405553e-01	 1.5449821e+00	 1.4313419e-01	[ 1.3196528e+00]	 2.0657516e-01


.. parsed-literal::

     147	 1.4912620e+00	 1.2401956e-01	 1.5467534e+00	 1.4299140e-01	  1.3192404e+00 	 2.0989919e-01


.. parsed-literal::

     148	 1.4930430e+00	 1.2401161e-01	 1.5485537e+00	 1.4294599e-01	[ 1.3208180e+00]	 2.1612692e-01


.. parsed-literal::

     149	 1.4944485e+00	 1.2401136e-01	 1.5499468e+00	 1.4301729e-01	  1.3164303e+00 	 2.0867801e-01
     150	 1.4957300e+00	 1.2401411e-01	 1.5512121e+00	 1.4318006e-01	  1.3162755e+00 	 2.0875764e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.14 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f8f88ed3b20>



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
    CPU times: user 1.75 s, sys: 34 ms, total: 1.78 s
    Wall time: 590 ms


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

