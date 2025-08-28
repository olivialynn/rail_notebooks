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
       1	-3.5368427e-01	 3.2297768e-01	-3.4406413e-01	 3.1057696e-01	[-3.2159469e-01]	 4.6084428e-01


.. parsed-literal::

       2	-2.7928369e-01	 3.1187299e-01	-2.5518678e-01	 2.9910214e-01	[-2.1738706e-01]	 2.3364162e-01


.. parsed-literal::

       3	-2.3112465e-01	 2.8954918e-01	-1.8784624e-01	 2.7513201e-01	[-1.3089442e-01]	 3.0094671e-01


.. parsed-literal::

       4	-1.9026593e-01	 2.7155055e-01	-1.4169060e-01	 2.5939505e-01	[-8.3307074e-02]	 2.9271865e-01


.. parsed-literal::

       5	-1.2932543e-01	 2.5851300e-01	-9.9170914e-02	 2.5045061e-01	[-5.6104570e-02]	 2.1443176e-01


.. parsed-literal::

       6	-7.7050611e-02	 2.5418963e-01	-4.9547176e-02	 2.4697824e-01	[-2.2632801e-02]	 2.1652293e-01


.. parsed-literal::

       7	-5.9159484e-02	 2.5039746e-01	-3.5266305e-02	 2.4303019e-01	[-7.4836490e-03]	 2.1848893e-01


.. parsed-literal::

       8	-4.6788670e-02	 2.4839643e-01	-2.6387293e-02	 2.4106168e-01	[ 3.5228219e-03]	 2.1318626e-01


.. parsed-literal::

       9	-3.3155228e-02	 2.4578975e-01	-1.5388518e-02	 2.3881686e-01	[ 1.1315842e-02]	 2.2420430e-01


.. parsed-literal::

      10	-2.3314864e-02	 2.4395604e-01	-7.5900477e-03	 2.3758538e-01	[ 1.9393579e-02]	 2.0787191e-01
      11	-1.8132600e-02	 2.4296623e-01	-3.1441392e-03	 2.3649882e-01	[ 2.4127741e-02]	 1.8794250e-01


.. parsed-literal::

      12	-1.3254809e-02	 2.4215220e-01	 9.0332626e-04	 2.3534127e-01	[ 2.9094192e-02]	 2.0255017e-01


.. parsed-literal::

      13	-8.7219077e-03	 2.4135732e-01	 5.0941924e-03	 2.3438930e-01	[ 3.3929085e-02]	 2.1330762e-01


.. parsed-literal::

      14	 7.8719282e-02	 2.2802875e-01	 1.0003264e-01	 2.2041087e-01	[ 1.2428496e-01]	 4.3681574e-01
      15	 1.3618717e-01	 2.2290154e-01	 1.5861418e-01	 2.1535209e-01	[ 1.8648014e-01]	 1.9817996e-01


.. parsed-literal::

      16	 2.2468747e-01	 2.1709893e-01	 2.5343557e-01	 2.0830752e-01	[ 3.0661704e-01]	 2.1605444e-01


.. parsed-literal::

      17	 2.6575181e-01	 2.1948455e-01	 2.9707334e-01	 2.0986208e-01	[ 3.4879058e-01]	 2.0601964e-01


.. parsed-literal::

      18	 3.0510294e-01	 2.1676573e-01	 3.3634903e-01	 2.0871223e-01	[ 3.8022617e-01]	 2.0677090e-01


.. parsed-literal::

      19	 3.4137360e-01	 2.1442571e-01	 3.7308342e-01	 2.0812276e-01	[ 4.1038122e-01]	 2.0764756e-01
      20	 4.0898629e-01	 2.1131993e-01	 4.4224935e-01	 2.0670831e-01	[ 4.7912510e-01]	 1.9086504e-01


.. parsed-literal::

      21	 4.9261112e-01	 2.1198615e-01	 5.2931442e-01	 2.0599960e-01	[ 5.8079506e-01]	 2.1632051e-01


.. parsed-literal::

      22	 5.4720380e-01	 2.1143227e-01	 5.8509912e-01	 2.0627142e-01	[ 6.3095514e-01]	 2.0417929e-01


.. parsed-literal::

      23	 5.8633683e-01	 2.0876750e-01	 6.2506435e-01	 2.0342529e-01	[ 6.7281685e-01]	 2.1031046e-01
      24	 6.1927377e-01	 2.0607935e-01	 6.5730832e-01	 2.0146021e-01	[ 7.0185347e-01]	 1.9861507e-01


.. parsed-literal::

      25	 6.5665890e-01	 2.0449045e-01	 6.9464979e-01	 2.0037075e-01	[ 7.3391467e-01]	 2.1853065e-01


.. parsed-literal::

      26	 6.8203476e-01	 2.0881304e-01	 7.1849478e-01	 2.0104387e-01	[ 7.6903000e-01]	 2.1763968e-01


.. parsed-literal::

      27	 7.2417144e-01	 2.0535281e-01	 7.6085024e-01	 2.0045164e-01	[ 7.9330967e-01]	 2.0558333e-01
      28	 7.5143531e-01	 2.0500782e-01	 7.9063580e-01	 2.0002091e-01	[ 8.1964039e-01]	 1.9765711e-01


.. parsed-literal::

      29	 7.8053155e-01	 2.0519757e-01	 8.2006937e-01	 2.0118583e-01	[ 8.4680665e-01]	 1.9266963e-01
      30	 7.9321339e-01	 2.1685260e-01	 8.3429583e-01	 2.1388133e-01	[ 8.4949116e-01]	 1.9864440e-01


.. parsed-literal::

      31	 8.2463504e-01	 2.0922669e-01	 8.6566652e-01	 2.0610300e-01	[ 8.8514058e-01]	 2.0862842e-01


.. parsed-literal::

      32	 8.4398436e-01	 2.0563812e-01	 8.8551192e-01	 2.0178197e-01	[ 9.0429251e-01]	 2.1858716e-01


.. parsed-literal::

      33	 8.6931144e-01	 2.0323385e-01	 9.1085745e-01	 1.9844329e-01	[ 9.2777530e-01]	 2.0844126e-01
      34	 8.8821885e-01	 2.0140614e-01	 9.3277827e-01	 1.9428547e-01	[ 9.5502232e-01]	 2.0081925e-01


.. parsed-literal::

      35	 9.2062068e-01	 1.9847356e-01	 9.6396804e-01	 1.9151980e-01	[ 9.8546820e-01]	 2.1681833e-01


.. parsed-literal::

      36	 9.3141293e-01	 1.9555490e-01	 9.7454011e-01	 1.8899070e-01	[ 9.9726730e-01]	 2.1469164e-01


.. parsed-literal::

      37	 9.4403255e-01	 1.9315207e-01	 9.8765558e-01	 1.8672356e-01	[ 1.0097078e+00]	 2.0530391e-01
      38	 9.6156869e-01	 1.9041612e-01	 1.0061030e+00	 1.8372243e-01	[ 1.0271967e+00]	 1.7883706e-01


.. parsed-literal::

      39	 9.8071821e-01	 1.8715317e-01	 1.0266122e+00	 1.8083442e-01	[ 1.0468833e+00]	 1.9094253e-01
      40	 9.9641322e-01	 1.8516464e-01	 1.0425192e+00	 1.7802367e-01	[ 1.0629216e+00]	 1.9697547e-01


.. parsed-literal::

      41	 1.0064263e+00	 1.8424964e-01	 1.0524201e+00	 1.7750465e-01	[ 1.0732534e+00]	 1.8805718e-01
      42	 1.0247207e+00	 1.8277664e-01	 1.0712793e+00	 1.7747639e-01	[ 1.0890954e+00]	 1.8543577e-01


.. parsed-literal::

      43	 1.0366652e+00	 1.8079745e-01	 1.0839214e+00	 1.7651953e-01	[ 1.0984103e+00]	 2.0380712e-01


.. parsed-literal::

      44	 1.0486597e+00	 1.7918532e-01	 1.0962715e+00	 1.7598117e-01	[ 1.1047004e+00]	 2.1502495e-01
      45	 1.0612019e+00	 1.7809238e-01	 1.1085884e+00	 1.7530268e-01	[ 1.1165899e+00]	 1.9933128e-01


.. parsed-literal::

      46	 1.0766250e+00	 1.7379637e-01	 1.1236732e+00	 1.7157993e-01	[ 1.1308992e+00]	 1.7956042e-01
      47	 1.0848292e+00	 1.7092492e-01	 1.1321040e+00	 1.6963069e-01	[ 1.1390846e+00]	 1.8752718e-01


.. parsed-literal::

      48	 1.0966875e+00	 1.6897153e-01	 1.1439092e+00	 1.6780934e-01	[ 1.1497261e+00]	 1.7687964e-01


.. parsed-literal::

      49	 1.1045777e+00	 1.6724982e-01	 1.1521858e+00	 1.6662365e-01	[ 1.1575917e+00]	 2.0148516e-01
      50	 1.1130636e+00	 1.6564915e-01	 1.1611220e+00	 1.6569549e-01	[ 1.1655449e+00]	 1.8493748e-01


.. parsed-literal::

      51	 1.1247426e+00	 1.6418831e-01	 1.1734061e+00	 1.6479569e-01	[ 1.1729924e+00]	 2.1183491e-01


.. parsed-literal::

      52	 1.1358392e+00	 1.6292633e-01	 1.1846668e+00	 1.6422566e-01	[ 1.1806546e+00]	 2.0786023e-01
      53	 1.1422986e+00	 1.6260155e-01	 1.1910422e+00	 1.6371216e-01	[ 1.1835943e+00]	 1.9777799e-01


.. parsed-literal::

      54	 1.1539859e+00	 1.6162446e-01	 1.2031294e+00	 1.6274029e-01	[ 1.1869988e+00]	 1.7468834e-01


.. parsed-literal::

      55	 1.1651674e+00	 1.6002955e-01	 1.2140837e+00	 1.6040853e-01	[ 1.1980477e+00]	 2.1205354e-01
      56	 1.1728206e+00	 1.5925167e-01	 1.2219116e+00	 1.5976637e-01	[ 1.2067474e+00]	 1.8100691e-01


.. parsed-literal::

      57	 1.1843129e+00	 1.5739606e-01	 1.2342441e+00	 1.5835361e-01	[ 1.2218600e+00]	 1.8772435e-01
      58	 1.1928757e+00	 1.5656466e-01	 1.2431033e+00	 1.5788977e-01	[ 1.2306346e+00]	 1.7263317e-01


.. parsed-literal::

      59	 1.2043635e+00	 1.5496685e-01	 1.2550164e+00	 1.5690016e-01	[ 1.2432840e+00]	 2.0102167e-01


.. parsed-literal::

      60	 1.2119966e+00	 1.5442760e-01	 1.2626323e+00	 1.5597157e-01	[ 1.2518354e+00]	 2.0551777e-01


.. parsed-literal::

      61	 1.2187697e+00	 1.5411830e-01	 1.2692059e+00	 1.5550410e-01	[ 1.2579976e+00]	 2.0527577e-01
      62	 1.2332364e+00	 1.5297655e-01	 1.2840141e+00	 1.5325036e-01	[ 1.2686451e+00]	 1.9980216e-01


.. parsed-literal::

      63	 1.2421116e+00	 1.5370642e-01	 1.2926961e+00	 1.5348883e-01	[ 1.2780756e+00]	 1.7681146e-01
      64	 1.2524323e+00	 1.5247539e-01	 1.3030718e+00	 1.5145022e-01	[ 1.2891322e+00]	 1.6889048e-01


.. parsed-literal::

      65	 1.2606698e+00	 1.5148577e-01	 1.3117197e+00	 1.5044990e-01	[ 1.2930780e+00]	 1.8351865e-01


.. parsed-literal::

      66	 1.2690117e+00	 1.5070939e-01	 1.3206199e+00	 1.4931613e-01	[ 1.2971096e+00]	 2.1762919e-01
      67	 1.2760426e+00	 1.5012785e-01	 1.3278622e+00	 1.4815187e-01	[ 1.3025843e+00]	 1.7731094e-01


.. parsed-literal::

      68	 1.2838212e+00	 1.4929587e-01	 1.3355677e+00	 1.4659902e-01	[ 1.3083651e+00]	 2.0635438e-01
      69	 1.2923580e+00	 1.4833042e-01	 1.3441822e+00	 1.4468189e-01	[ 1.3121469e+00]	 1.9864345e-01


.. parsed-literal::

      70	 1.3003495e+00	 1.4756247e-01	 1.3524042e+00	 1.4299089e-01	[ 1.3153960e+00]	 2.0368719e-01


.. parsed-literal::

      71	 1.3091178e+00	 1.4656891e-01	 1.3614449e+00	 1.4136352e-01	[ 1.3165856e+00]	 2.1080089e-01


.. parsed-literal::

      72	 1.3154763e+00	 1.4550590e-01	 1.3679140e+00	 1.4036165e-01	[ 1.3199473e+00]	 2.0822787e-01


.. parsed-literal::

      73	 1.3204585e+00	 1.4436550e-01	 1.3730424e+00	 1.3951373e-01	[ 1.3234763e+00]	 2.1845770e-01


.. parsed-literal::

      74	 1.3267250e+00	 1.4364143e-01	 1.3793633e+00	 1.3908957e-01	[ 1.3284269e+00]	 2.1563697e-01


.. parsed-literal::

      75	 1.3341937e+00	 1.4225785e-01	 1.3869146e+00	 1.3819991e-01	[ 1.3309646e+00]	 2.1658802e-01
      76	 1.3406157e+00	 1.4212229e-01	 1.3931718e+00	 1.3796669e-01	[ 1.3382337e+00]	 1.8146253e-01


.. parsed-literal::

      77	 1.3460752e+00	 1.4170407e-01	 1.3985651e+00	 1.3715852e-01	[ 1.3451905e+00]	 1.8910217e-01


.. parsed-literal::

      78	 1.3519802e+00	 1.4111150e-01	 1.4046297e+00	 1.3652408e-01	[ 1.3533728e+00]	 2.0366573e-01


.. parsed-literal::

      79	 1.3582587e+00	 1.4047703e-01	 1.4110380e+00	 1.3584794e-01	[ 1.3593410e+00]	 2.1634245e-01


.. parsed-literal::

      80	 1.3632828e+00	 1.3996148e-01	 1.4160709e+00	 1.3585056e-01	[ 1.3652180e+00]	 2.1455956e-01


.. parsed-literal::

      81	 1.3718318e+00	 1.3933761e-01	 1.4249620e+00	 1.3628254e-01	[ 1.3663315e+00]	 2.0732617e-01
      82	 1.3774928e+00	 1.3885436e-01	 1.4307226e+00	 1.3655402e-01	[ 1.3724695e+00]	 1.8373442e-01


.. parsed-literal::

      83	 1.3830433e+00	 1.3855578e-01	 1.4361671e+00	 1.3617785e-01	[ 1.3731129e+00]	 2.0435882e-01


.. parsed-literal::

      84	 1.3872194e+00	 1.3827181e-01	 1.4403803e+00	 1.3555022e-01	  1.3699206e+00 	 2.0484281e-01
      85	 1.3914465e+00	 1.3784641e-01	 1.4447673e+00	 1.3470221e-01	  1.3695959e+00 	 1.9334674e-01


.. parsed-literal::

      86	 1.3959408e+00	 1.3726568e-01	 1.4495160e+00	 1.3366925e-01	  1.3580347e+00 	 2.0383310e-01


.. parsed-literal::

      87	 1.4003299e+00	 1.3685325e-01	 1.4537973e+00	 1.3327980e-01	  1.3655488e+00 	 2.0652437e-01


.. parsed-literal::

      88	 1.4034597e+00	 1.3652132e-01	 1.4568687e+00	 1.3319493e-01	  1.3711361e+00 	 2.1992612e-01


.. parsed-literal::

      89	 1.4077118e+00	 1.3609152e-01	 1.4611179e+00	 1.3284628e-01	  1.3716985e+00 	 2.1835923e-01


.. parsed-literal::

      90	 1.4122808e+00	 1.3517538e-01	 1.4658637e+00	 1.3229475e-01	  1.3639758e+00 	 2.1507764e-01
      91	 1.4171526e+00	 1.3488101e-01	 1.4707501e+00	 1.3194690e-01	  1.3601454e+00 	 1.8185830e-01


.. parsed-literal::

      92	 1.4205863e+00	 1.3472797e-01	 1.4742598e+00	 1.3179525e-01	  1.3570585e+00 	 2.1494770e-01


.. parsed-literal::

      93	 1.4245609e+00	 1.3440213e-01	 1.4783556e+00	 1.3164817e-01	  1.3508083e+00 	 2.1020603e-01


.. parsed-literal::

      94	 1.4278817e+00	 1.3411450e-01	 1.4817113e+00	 1.3149404e-01	  1.3457661e+00 	 2.1345282e-01


.. parsed-literal::

      95	 1.4322190e+00	 1.3380955e-01	 1.4859861e+00	 1.3141864e-01	  1.3441605e+00 	 2.1767712e-01


.. parsed-literal::

      96	 1.4349876e+00	 1.3342659e-01	 1.4887656e+00	 1.3119513e-01	  1.3358398e+00 	 2.0595431e-01


.. parsed-literal::

      97	 1.4377408e+00	 1.3326852e-01	 1.4915993e+00	 1.3094877e-01	  1.3281151e+00 	 2.0157313e-01


.. parsed-literal::

      98	 1.4398542e+00	 1.3302418e-01	 1.4940615e+00	 1.3084533e-01	  1.3154587e+00 	 2.1009660e-01
      99	 1.4435233e+00	 1.3294860e-01	 1.4976808e+00	 1.3052751e-01	  1.3100876e+00 	 1.9842601e-01


.. parsed-literal::

     100	 1.4454103e+00	 1.3294948e-01	 1.4995814e+00	 1.3049341e-01	  1.3102760e+00 	 2.1258783e-01


.. parsed-literal::

     101	 1.4485378e+00	 1.3291289e-01	 1.5027512e+00	 1.3050686e-01	  1.3079178e+00 	 2.1152115e-01


.. parsed-literal::

     102	 1.4533854e+00	 1.3286232e-01	 1.5076459e+00	 1.3055604e-01	  1.3017995e+00 	 2.0811176e-01
     103	 1.4547982e+00	 1.3301328e-01	 1.5093054e+00	 1.3086662e-01	  1.2991274e+00 	 1.9940281e-01


.. parsed-literal::

     104	 1.4590169e+00	 1.3282762e-01	 1.5133298e+00	 1.3064182e-01	  1.2995925e+00 	 2.1664166e-01
     105	 1.4604724e+00	 1.3268111e-01	 1.5147706e+00	 1.3045930e-01	  1.2994631e+00 	 1.7987943e-01


.. parsed-literal::

     106	 1.4637179e+00	 1.3227602e-01	 1.5181197e+00	 1.2996591e-01	  1.2836303e+00 	 2.1520758e-01


.. parsed-literal::

     107	 1.4665255e+00	 1.3180006e-01	 1.5209359e+00	 1.2933035e-01	  1.2862473e+00 	 2.1102595e-01


.. parsed-literal::

     108	 1.4688115e+00	 1.3168353e-01	 1.5231888e+00	 1.2915251e-01	  1.2857579e+00 	 2.0699024e-01


.. parsed-literal::

     109	 1.4716265e+00	 1.3145911e-01	 1.5260082e+00	 1.2894783e-01	  1.2818650e+00 	 2.9850030e-01


.. parsed-literal::

     110	 1.4740030e+00	 1.3129531e-01	 1.5284665e+00	 1.2894044e-01	  1.2764437e+00 	 2.0835686e-01


.. parsed-literal::

     111	 1.4769503e+00	 1.3101461e-01	 1.5314311e+00	 1.2881170e-01	  1.2687657e+00 	 2.0265746e-01


.. parsed-literal::

     112	 1.4795328e+00	 1.3072470e-01	 1.5341466e+00	 1.2884426e-01	  1.2553150e+00 	 2.1977425e-01


.. parsed-literal::

     113	 1.4814332e+00	 1.3066242e-01	 1.5360297e+00	 1.2880703e-01	  1.2510859e+00 	 2.1727967e-01
     114	 1.4831206e+00	 1.3054814e-01	 1.5377021e+00	 1.2865292e-01	  1.2477419e+00 	 1.7127490e-01


.. parsed-literal::

     115	 1.4861010e+00	 1.3035627e-01	 1.5407382e+00	 1.2839724e-01	  1.2415752e+00 	 2.0690680e-01
     116	 1.4868000e+00	 1.3003107e-01	 1.5416170e+00	 1.2794328e-01	  1.2161065e+00 	 1.9372129e-01


.. parsed-literal::

     117	 1.4889727e+00	 1.3005587e-01	 1.5436624e+00	 1.2796822e-01	  1.2303134e+00 	 1.9269466e-01


.. parsed-literal::

     118	 1.4904510e+00	 1.2998472e-01	 1.5451527e+00	 1.2789188e-01	  1.2337688e+00 	 2.0685863e-01
     119	 1.4919563e+00	 1.2985089e-01	 1.5467149e+00	 1.2771370e-01	  1.2318906e+00 	 1.9746947e-01


.. parsed-literal::

     120	 1.4946704e+00	 1.2964795e-01	 1.5495133e+00	 1.2739716e-01	  1.2259980e+00 	 2.1836877e-01


.. parsed-literal::

     121	 1.4962140e+00	 1.2954814e-01	 1.5513595e+00	 1.2691603e-01	  1.1996163e+00 	 2.0826268e-01


.. parsed-literal::

     122	 1.5001227e+00	 1.2947308e-01	 1.5551421e+00	 1.2671436e-01	  1.2132740e+00 	 2.0905066e-01


.. parsed-literal::

     123	 1.5013450e+00	 1.2948890e-01	 1.5562964e+00	 1.2673900e-01	  1.2173553e+00 	 2.1173048e-01


.. parsed-literal::

     124	 1.5034295e+00	 1.2946527e-01	 1.5584029e+00	 1.2665938e-01	  1.2156963e+00 	 2.1789002e-01
     125	 1.5044064e+00	 1.2927432e-01	 1.5595733e+00	 1.2625062e-01	  1.2182983e+00 	 1.7722774e-01


.. parsed-literal::

     126	 1.5068018e+00	 1.2929253e-01	 1.5618883e+00	 1.2631840e-01	  1.2121418e+00 	 2.1510744e-01


.. parsed-literal::

     127	 1.5078270e+00	 1.2920486e-01	 1.5629127e+00	 1.2625779e-01	  1.2099398e+00 	 2.0746017e-01


.. parsed-literal::

     128	 1.5093045e+00	 1.2906092e-01	 1.5643932e+00	 1.2613950e-01	  1.2080519e+00 	 2.1601605e-01
     129	 1.5116196e+00	 1.2887067e-01	 1.5667231e+00	 1.2587161e-01	  1.2018249e+00 	 2.0636797e-01


.. parsed-literal::

     130	 1.5129727e+00	 1.2869498e-01	 1.5681248e+00	 1.2568628e-01	  1.1988735e+00 	 3.2413483e-01


.. parsed-literal::

     131	 1.5149161e+00	 1.2856678e-01	 1.5700627e+00	 1.2545286e-01	  1.1930442e+00 	 2.0784903e-01
     132	 1.5163050e+00	 1.2848650e-01	 1.5714725e+00	 1.2524893e-01	  1.1871659e+00 	 1.9435859e-01


.. parsed-literal::

     133	 1.5176918e+00	 1.2838161e-01	 1.5728962e+00	 1.2515897e-01	  1.1818993e+00 	 1.9813371e-01


.. parsed-literal::

     134	 1.5190938e+00	 1.2825076e-01	 1.5743422e+00	 1.2500327e-01	  1.1752328e+00 	 2.1798301e-01


.. parsed-literal::

     135	 1.5204029e+00	 1.2816887e-01	 1.5756614e+00	 1.2499893e-01	  1.1756220e+00 	 2.0298934e-01


.. parsed-literal::

     136	 1.5219926e+00	 1.2803434e-01	 1.5772451e+00	 1.2504497e-01	  1.1750398e+00 	 2.0429158e-01
     137	 1.5236964e+00	 1.2789076e-01	 1.5789391e+00	 1.2500323e-01	  1.1758423e+00 	 1.8644404e-01


.. parsed-literal::

     138	 1.5255094e+00	 1.2771473e-01	 1.5807467e+00	 1.2482079e-01	  1.1771519e+00 	 2.1346736e-01
     139	 1.5271663e+00	 1.2739020e-01	 1.5824637e+00	 1.2448415e-01	  1.1675400e+00 	 1.9786143e-01


.. parsed-literal::

     140	 1.5286162e+00	 1.2712258e-01	 1.5839997e+00	 1.2412310e-01	  1.1699495e+00 	 2.1092963e-01


.. parsed-literal::

     141	 1.5300608e+00	 1.2702947e-01	 1.5854516e+00	 1.2393321e-01	  1.1657835e+00 	 2.0647693e-01
     142	 1.5312503e+00	 1.2688721e-01	 1.5866853e+00	 1.2384116e-01	  1.1611774e+00 	 1.8241692e-01


.. parsed-literal::

     143	 1.5325081e+00	 1.2673591e-01	 1.5879857e+00	 1.2385382e-01	  1.1557340e+00 	 1.8906856e-01


.. parsed-literal::

     144	 1.5338475e+00	 1.2647940e-01	 1.5894242e+00	 1.2394543e-01	  1.1591676e+00 	 2.0908976e-01
     145	 1.5354109e+00	 1.2640480e-01	 1.5909106e+00	 1.2400159e-01	  1.1545251e+00 	 1.7982912e-01


.. parsed-literal::

     146	 1.5361581e+00	 1.2636305e-01	 1.5916060e+00	 1.2394864e-01	  1.1558207e+00 	 1.8925190e-01


.. parsed-literal::

     147	 1.5376336e+00	 1.2623204e-01	 1.5930348e+00	 1.2372943e-01	  1.1551848e+00 	 2.1307206e-01


.. parsed-literal::

     148	 1.5394906e+00	 1.2605942e-01	 1.5949237e+00	 1.2332931e-01	  1.1487782e+00 	 2.1347165e-01


.. parsed-literal::

     149	 1.5404716e+00	 1.2591115e-01	 1.5959565e+00	 1.2289068e-01	  1.1395704e+00 	 3.1451321e-01


.. parsed-literal::

     150	 1.5416759e+00	 1.2584033e-01	 1.5972301e+00	 1.2259876e-01	  1.1331209e+00 	 2.1177149e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.1 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f52f0b6d7b0>



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
    CPU times: user 1.99 s, sys: 40.9 ms, total: 2.03 s
    Wall time: 597 ms


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

