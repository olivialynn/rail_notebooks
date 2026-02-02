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
       1	-3.3237434e-01	 3.1743778e-01	-3.2260121e-01	 3.3207822e-01	[-3.5178992e-01]	 4.4566369e-01


.. parsed-literal::

       2	-2.5962462e-01	 3.0531392e-01	-2.3372095e-01	 3.2256284e-01	[-2.8851180e-01]	 2.2413111e-01


.. parsed-literal::

       3	-2.1807244e-01	 2.8644850e-01	-1.7558691e-01	 3.0768405e-01	[-2.6878911e-01]	 2.6703668e-01
       4	-1.7735886e-01	 2.6085162e-01	-1.3274000e-01	 2.8662144e-01	 -2.9504354e-01 	 1.7648458e-01


.. parsed-literal::

       5	-9.1061187e-02	 2.5209149e-01	-5.4431764e-02	 2.7436830e-01	[-1.7362215e-01]	 1.7164707e-01
       6	-5.0832127e-02	 2.4616307e-01	-1.8130724e-02	 2.6986149e-01	[-1.1344480e-01]	 1.8024850e-01


.. parsed-literal::

       7	-3.3922511e-02	 2.4375979e-01	-7.6700289e-03	 2.6768046e-01	[-1.0972997e-01]	 1.9742465e-01


.. parsed-literal::

       8	-1.9539796e-02	 2.4148013e-01	 2.0033230e-03	 2.6586649e-01	[-1.0420441e-01]	 2.0177841e-01
       9	-6.2833692e-03	 2.3919069e-01	 1.1920483e-02	 2.6420881e-01	[-9.9676124e-02]	 1.9519138e-01


.. parsed-literal::

      10	 5.1514775e-03	 2.3718101e-01	 2.0840134e-02	 2.6233511e-01	[-8.7948386e-02]	 2.1749330e-01


.. parsed-literal::

      11	 8.6955900e-03	 2.3661712e-01	 2.2906622e-02	 2.6172787e-01	 -9.6907997e-02 	 2.0938826e-01


.. parsed-literal::

      12	 1.2560510e-02	 2.3602907e-01	 2.6657814e-02	 2.6131473e-01	[-8.4962705e-02]	 2.0172071e-01
      13	 1.4977613e-02	 2.3551473e-01	 2.9031701e-02	 2.6077642e-01	[-8.2640270e-02]	 1.7991972e-01


.. parsed-literal::

      14	 1.8436969e-02	 2.3485619e-01	 3.2544253e-02	 2.6022365e-01	[-7.8772440e-02]	 1.8483663e-01


.. parsed-literal::

      15	 1.2617364e-01	 2.2073390e-01	 1.4563343e-01	 2.4720998e-01	[ 4.2712154e-02]	 3.1998777e-01
      16	 1.5156658e-01	 2.1907058e-01	 1.7429841e-01	 2.4372103e-01	[ 1.1113364e-01]	 1.9879246e-01


.. parsed-literal::

      17	 2.6649505e-01	 2.1237893e-01	 2.9194903e-01	 2.3998077e-01	[ 1.9282472e-01]	 2.0567799e-01
      18	 3.2802809e-01	 2.0504048e-01	 3.5935662e-01	 2.3265784e-01	  1.7793043e-01 	 1.8683743e-01


.. parsed-literal::

      19	 3.8314996e-01	 2.0145966e-01	 4.1492114e-01	 2.2760842e-01	[ 2.7436729e-01]	 1.6292548e-01


.. parsed-literal::

      20	 4.2432223e-01	 1.9877455e-01	 4.5560505e-01	 2.2553802e-01	[ 3.3185774e-01]	 2.1257401e-01
      21	 4.9908693e-01	 1.9542534e-01	 5.3110393e-01	 2.2027411e-01	[ 4.2129241e-01]	 1.8706298e-01


.. parsed-literal::

      22	 6.4322468e-01	 1.9207535e-01	 6.7942758e-01	 2.2099166e-01	[ 5.4348139e-01]	 2.0724201e-01
      23	 6.4705638e-01	 1.9431001e-01	 6.8928749e-01	 2.2375515e-01	  5.2258271e-01 	 1.7508698e-01


.. parsed-literal::

      24	 7.1712693e-01	 1.9026256e-01	 7.5462230e-01	 2.1791506e-01	[ 6.3217226e-01]	 1.9627810e-01
      25	 7.4090830e-01	 1.8507362e-01	 7.7766084e-01	 2.1313844e-01	[ 6.5405112e-01]	 1.7323184e-01


.. parsed-literal::

      26	 7.5218445e-01	 1.8736184e-01	 7.8847310e-01	 2.1397929e-01	  6.4117935e-01 	 1.6071749e-01


.. parsed-literal::

      27	 7.8684487e-01	 1.8325865e-01	 8.2349492e-01	 2.1097988e-01	[ 6.6985406e-01]	 2.1355510e-01
      28	 8.0654844e-01	 1.8024530e-01	 8.4393125e-01	 2.0787085e-01	[ 6.8832937e-01]	 1.6688800e-01


.. parsed-literal::

      29	 8.2824291e-01	 1.7803151e-01	 8.6631109e-01	 2.0596528e-01	[ 7.0109685e-01]	 1.9515705e-01


.. parsed-literal::

      30	 8.5751763e-01	 1.7564615e-01	 8.9599744e-01	 2.0380519e-01	[ 7.2515357e-01]	 2.0425200e-01
      31	 8.8343654e-01	 1.7462366e-01	 9.2213176e-01	 2.0382973e-01	[ 7.4999249e-01]	 1.9316483e-01


.. parsed-literal::

      32	 9.1157657e-01	 1.7138995e-01	 9.5144051e-01	 2.0046553e-01	[ 7.7929480e-01]	 2.0565677e-01


.. parsed-literal::

      33	 9.3390636e-01	 1.6983586e-01	 9.7444262e-01	 1.9978119e-01	[ 7.8817367e-01]	 2.1015191e-01
      34	 9.5417271e-01	 1.6668795e-01	 9.9604801e-01	 1.9642397e-01	[ 7.9057786e-01]	 1.7261004e-01


.. parsed-literal::

      35	 9.7302230e-01	 1.6407744e-01	 1.0152479e+00	 1.9356629e-01	[ 8.1470088e-01]	 2.0412540e-01


.. parsed-literal::

      36	 9.8882691e-01	 1.6177039e-01	 1.0312049e+00	 1.9161809e-01	[ 8.3194208e-01]	 2.1325421e-01


.. parsed-literal::

      37	 1.0051493e+00	 1.6108024e-01	 1.0476897e+00	 1.8993482e-01	[ 8.6178804e-01]	 2.1229315e-01
      38	 1.0137972e+00	 1.5918318e-01	 1.0564618e+00	 1.9018931e-01	[ 8.7117801e-01]	 1.9987774e-01


.. parsed-literal::

      39	 1.0238160e+00	 1.5832585e-01	 1.0663093e+00	 1.8959779e-01	[ 8.8381895e-01]	 1.8851709e-01


.. parsed-literal::

      40	 1.0352680e+00	 1.5717997e-01	 1.0779305e+00	 1.8891706e-01	[ 8.9809363e-01]	 2.0842743e-01


.. parsed-literal::

      41	 1.0446565e+00	 1.5622469e-01	 1.0877096e+00	 1.8811989e-01	[ 9.0631490e-01]	 2.0585251e-01


.. parsed-literal::

      42	 1.0662425e+00	 1.5512265e-01	 1.1108246e+00	 1.8721146e-01	[ 9.3143700e-01]	 2.0283628e-01
      43	 1.0692553e+00	 1.5550759e-01	 1.1146345e+00	 1.8720519e-01	[ 9.3660300e-01]	 2.0013595e-01


.. parsed-literal::

      44	 1.0814022e+00	 1.5355675e-01	 1.1259758e+00	 1.8552211e-01	[ 9.4870185e-01]	 2.0550537e-01
      45	 1.0871622e+00	 1.5278750e-01	 1.1318970e+00	 1.8483959e-01	[ 9.5183820e-01]	 1.8838477e-01


.. parsed-literal::

      46	 1.0930702e+00	 1.5196963e-01	 1.1381494e+00	 1.8413298e-01	[ 9.5499162e-01]	 2.0661616e-01
      47	 1.1054805e+00	 1.5048864e-01	 1.1516183e+00	 1.8277048e-01	[ 9.6021379e-01]	 1.9754195e-01


.. parsed-literal::

      48	 1.1109998e+00	 1.4863345e-01	 1.1577509e+00	 1.8055667e-01	[ 9.6743680e-01]	 1.9887972e-01


.. parsed-literal::

      49	 1.1205130e+00	 1.4791959e-01	 1.1669251e+00	 1.7994003e-01	[ 9.7811202e-01]	 2.1701622e-01


.. parsed-literal::

      50	 1.1301536e+00	 1.4659084e-01	 1.1765599e+00	 1.7905005e-01	[ 9.8863888e-01]	 2.1969914e-01
      51	 1.1389201e+00	 1.4528586e-01	 1.1853269e+00	 1.7809143e-01	[ 1.0004202e+00]	 1.7829347e-01


.. parsed-literal::

      52	 1.1509879e+00	 1.4398832e-01	 1.1977741e+00	 1.7790966e-01	[ 1.0094572e+00]	 1.8277240e-01


.. parsed-literal::

      53	 1.1587484e+00	 1.4315959e-01	 1.2058049e+00	 1.7748340e-01	[ 1.0274787e+00]	 2.0357108e-01


.. parsed-literal::

      54	 1.1652893e+00	 1.4291173e-01	 1.2122268e+00	 1.7774757e-01	[ 1.0279149e+00]	 2.0335841e-01


.. parsed-literal::

      55	 1.1725141e+00	 1.4262567e-01	 1.2197567e+00	 1.7812306e-01	[ 1.0287862e+00]	 2.1393561e-01
      56	 1.1795403e+00	 1.4226652e-01	 1.2269049e+00	 1.7812248e-01	[ 1.0363737e+00]	 1.7158699e-01


.. parsed-literal::

      57	 1.1898818e+00	 1.4194175e-01	 1.2377119e+00	 1.7820474e-01	[ 1.0543808e+00]	 1.8347692e-01


.. parsed-literal::

      58	 1.1997454e+00	 1.4074852e-01	 1.2475485e+00	 1.7789286e-01	[ 1.0656397e+00]	 2.0593882e-01


.. parsed-literal::

      59	 1.2062263e+00	 1.4002637e-01	 1.2540713e+00	 1.7715011e-01	[ 1.0732139e+00]	 2.0875502e-01


.. parsed-literal::

      60	 1.2132417e+00	 1.3922356e-01	 1.2614339e+00	 1.7706003e-01	[ 1.0801362e+00]	 2.1459556e-01
      61	 1.2197972e+00	 1.3878231e-01	 1.2683265e+00	 1.7612227e-01	[ 1.0868360e+00]	 1.8079066e-01


.. parsed-literal::

      62	 1.2254353e+00	 1.3841855e-01	 1.2740973e+00	 1.7552986e-01	[ 1.0923940e+00]	 2.1722794e-01


.. parsed-literal::

      63	 1.2331731e+00	 1.3804088e-01	 1.2822555e+00	 1.7516827e-01	[ 1.0951065e+00]	 2.1656513e-01
      64	 1.2395632e+00	 1.3746673e-01	 1.2888298e+00	 1.7425920e-01	[ 1.1048802e+00]	 1.8040419e-01


.. parsed-literal::

      65	 1.2459373e+00	 1.3669321e-01	 1.2951579e+00	 1.7328795e-01	[ 1.1101838e+00]	 2.0041704e-01


.. parsed-literal::

      66	 1.2545378e+00	 1.3582184e-01	 1.3040300e+00	 1.7189955e-01	[ 1.1188734e+00]	 2.0256186e-01


.. parsed-literal::

      67	 1.2605059e+00	 1.3506324e-01	 1.3101566e+00	 1.7067606e-01	[ 1.1273584e+00]	 2.0069528e-01


.. parsed-literal::

      68	 1.2675647e+00	 1.3433937e-01	 1.3173922e+00	 1.6986147e-01	[ 1.1361934e+00]	 2.1150184e-01
      69	 1.2727348e+00	 1.3326336e-01	 1.3226507e+00	 1.6859488e-01	[ 1.1388433e+00]	 1.9127297e-01


.. parsed-literal::

      70	 1.2780027e+00	 1.3292599e-01	 1.3280329e+00	 1.6834058e-01	[ 1.1407713e+00]	 1.9946814e-01


.. parsed-literal::

      71	 1.2851484e+00	 1.3262214e-01	 1.3351844e+00	 1.6774411e-01	[ 1.1415358e+00]	 2.1737504e-01
      72	 1.2895618e+00	 1.3247848e-01	 1.3397637e+00	 1.6783045e-01	[ 1.1421111e+00]	 1.9752789e-01


.. parsed-literal::

      73	 1.2959782e+00	 1.3233788e-01	 1.3459177e+00	 1.6807199e-01	[ 1.1518113e+00]	 1.9828033e-01
      74	 1.2989005e+00	 1.3216860e-01	 1.3488806e+00	 1.6797329e-01	[ 1.1567860e+00]	 1.7916250e-01


.. parsed-literal::

      75	 1.3047331e+00	 1.3190976e-01	 1.3549982e+00	 1.6783331e-01	[ 1.1624869e+00]	 1.8768239e-01


.. parsed-literal::

      76	 1.3097143e+00	 1.3187610e-01	 1.3604826e+00	 1.6764765e-01	[ 1.1687635e+00]	 2.1010900e-01


.. parsed-literal::

      77	 1.3151176e+00	 1.3111219e-01	 1.3659187e+00	 1.6668470e-01	[ 1.1700182e+00]	 2.0322514e-01
      78	 1.3203343e+00	 1.3017249e-01	 1.3712755e+00	 1.6541853e-01	  1.1682399e+00 	 1.7831588e-01


.. parsed-literal::

      79	 1.3242421e+00	 1.2972530e-01	 1.3752003e+00	 1.6484109e-01	  1.1696037e+00 	 1.7872095e-01
      80	 1.3293668e+00	 1.2905224e-01	 1.3806559e+00	 1.6408944e-01	  1.1672594e+00 	 1.9916773e-01


.. parsed-literal::

      81	 1.3366412e+00	 1.2861349e-01	 1.3877419e+00	 1.6314921e-01	[ 1.1774859e+00]	 2.1298361e-01
      82	 1.3397552e+00	 1.2854813e-01	 1.3908549e+00	 1.6305645e-01	[ 1.1843092e+00]	 1.7697859e-01


.. parsed-literal::

      83	 1.3474683e+00	 1.2825987e-01	 1.3989706e+00	 1.6227972e-01	[ 1.1958894e+00]	 2.0013976e-01


.. parsed-literal::

      84	 1.3510355e+00	 1.2832462e-01	 1.4028660e+00	 1.6176951e-01	[ 1.2068388e+00]	 2.2255421e-01


.. parsed-literal::

      85	 1.3560041e+00	 1.2793850e-01	 1.4077128e+00	 1.6135489e-01	[ 1.2081174e+00]	 2.1549225e-01
      86	 1.3602308e+00	 1.2752929e-01	 1.4118830e+00	 1.6079275e-01	[ 1.2097567e+00]	 1.8531823e-01


.. parsed-literal::

      87	 1.3636100e+00	 1.2733112e-01	 1.4152782e+00	 1.6046637e-01	[ 1.2125081e+00]	 2.0093155e-01
      88	 1.3708186e+00	 1.2666572e-01	 1.4226582e+00	 1.5927089e-01	[ 1.2217850e+00]	 1.9869757e-01


.. parsed-literal::

      89	 1.3745881e+00	 1.2654994e-01	 1.4267678e+00	 1.5893331e-01	[ 1.2222762e+00]	 1.7208576e-01
      90	 1.3795430e+00	 1.2644404e-01	 1.4315752e+00	 1.5883448e-01	[ 1.2290772e+00]	 1.7377567e-01


.. parsed-literal::

      91	 1.3830045e+00	 1.2616198e-01	 1.4350824e+00	 1.5843421e-01	[ 1.2312819e+00]	 1.7573261e-01
      92	 1.3876480e+00	 1.2580036e-01	 1.4399720e+00	 1.5785669e-01	[ 1.2314261e+00]	 1.7964077e-01


.. parsed-literal::

      93	 1.3933168e+00	 1.2518301e-01	 1.4459062e+00	 1.5663700e-01	  1.2301045e+00 	 1.8965316e-01
      94	 1.3982992e+00	 1.2475228e-01	 1.4509788e+00	 1.5615592e-01	[ 1.2330908e+00]	 2.0542192e-01


.. parsed-literal::

      95	 1.4032183e+00	 1.2443350e-01	 1.4558956e+00	 1.5591435e-01	[ 1.2419225e+00]	 1.7186499e-01


.. parsed-literal::

      96	 1.4068714e+00	 1.2425250e-01	 1.4595499e+00	 1.5583191e-01	[ 1.2456820e+00]	 2.0808196e-01


.. parsed-literal::

      97	 1.4111811e+00	 1.2402157e-01	 1.4638922e+00	 1.5567565e-01	[ 1.2476118e+00]	 2.0897365e-01


.. parsed-literal::

      98	 1.4153149e+00	 1.2377752e-01	 1.4681927e+00	 1.5530422e-01	[ 1.2476620e+00]	 2.0186400e-01


.. parsed-literal::

      99	 1.4189714e+00	 1.2341862e-01	 1.4718731e+00	 1.5475764e-01	  1.2452410e+00 	 2.0603418e-01


.. parsed-literal::

     100	 1.4222463e+00	 1.2317450e-01	 1.4751508e+00	 1.5447084e-01	[ 1.2476622e+00]	 2.1411228e-01
     101	 1.4266221e+00	 1.2266004e-01	 1.4796174e+00	 1.5376755e-01	[ 1.2492030e+00]	 1.9899321e-01


.. parsed-literal::

     102	 1.4290845e+00	 1.2254116e-01	 1.4822430e+00	 1.5364280e-01	  1.2456802e+00 	 1.7743158e-01


.. parsed-literal::

     103	 1.4315933e+00	 1.2240663e-01	 1.4846950e+00	 1.5345179e-01	  1.2489743e+00 	 2.1116471e-01


.. parsed-literal::

     104	 1.4350555e+00	 1.2217369e-01	 1.4882591e+00	 1.5289142e-01	  1.2483009e+00 	 2.0587754e-01
     105	 1.4375518e+00	 1.2198394e-01	 1.4908478e+00	 1.5254542e-01	  1.2488609e+00 	 1.9902372e-01


.. parsed-literal::

     106	 1.4407809e+00	 1.2182919e-01	 1.4941935e+00	 1.5241244e-01	  1.2478828e+00 	 1.9337225e-01


.. parsed-literal::

     107	 1.4437836e+00	 1.2146225e-01	 1.4973069e+00	 1.5205832e-01	  1.2459216e+00 	 2.0668888e-01
     108	 1.4463457e+00	 1.2112964e-01	 1.4999053e+00	 1.5206399e-01	  1.2450763e+00 	 1.7743158e-01


.. parsed-literal::

     109	 1.4486247e+00	 1.2089484e-01	 1.5021863e+00	 1.5217003e-01	  1.2457925e+00 	 1.8302298e-01


.. parsed-literal::

     110	 1.4516577e+00	 1.2059851e-01	 1.5052579e+00	 1.5209855e-01	  1.2470608e+00 	 2.1345377e-01


.. parsed-literal::

     111	 1.4549117e+00	 1.2027952e-01	 1.5086637e+00	 1.5250850e-01	  1.2448094e+00 	 2.0611382e-01


.. parsed-literal::

     112	 1.4580880e+00	 1.2011673e-01	 1.5118914e+00	 1.5285792e-01	  1.2439501e+00 	 2.0906425e-01
     113	 1.4605913e+00	 1.2004853e-01	 1.5144254e+00	 1.5303564e-01	  1.2435188e+00 	 1.9614673e-01


.. parsed-literal::

     114	 1.4639184e+00	 1.1981944e-01	 1.5178511e+00	 1.5340469e-01	  1.2396415e+00 	 2.0069265e-01


.. parsed-literal::

     115	 1.4661418e+00	 1.1971508e-01	 1.5201500e+00	 1.5372976e-01	  1.2398887e+00 	 2.8491712e-01
     116	 1.4691777e+00	 1.1945921e-01	 1.5232248e+00	 1.5375306e-01	  1.2388161e+00 	 1.9167900e-01


.. parsed-literal::

     117	 1.4722722e+00	 1.1907650e-01	 1.5264024e+00	 1.5374693e-01	  1.2377723e+00 	 1.8142533e-01


.. parsed-literal::

     118	 1.4735338e+00	 1.1884097e-01	 1.5277436e+00	 1.5374746e-01	  1.2358551e+00 	 2.0856237e-01
     119	 1.4755060e+00	 1.1874821e-01	 1.5296353e+00	 1.5363861e-01	  1.2397207e+00 	 1.8937564e-01


.. parsed-literal::

     120	 1.4769283e+00	 1.1864195e-01	 1.5310286e+00	 1.5364928e-01	  1.2411959e+00 	 1.7455244e-01
     121	 1.4788021e+00	 1.1847206e-01	 1.5328977e+00	 1.5372308e-01	  1.2410938e+00 	 1.7649841e-01


.. parsed-literal::

     122	 1.4816462e+00	 1.1826235e-01	 1.5357680e+00	 1.5372689e-01	  1.2417987e+00 	 1.7411232e-01


.. parsed-literal::

     123	 1.4833908e+00	 1.1811578e-01	 1.5375880e+00	 1.5399235e-01	  1.2404170e+00 	 3.2052803e-01
     124	 1.4860894e+00	 1.1795811e-01	 1.5403523e+00	 1.5394618e-01	  1.2411278e+00 	 1.7800546e-01


.. parsed-literal::

     125	 1.4882563e+00	 1.1787764e-01	 1.5425704e+00	 1.5384985e-01	  1.2426284e+00 	 2.1517467e-01
     126	 1.4906239e+00	 1.1772097e-01	 1.5450237e+00	 1.5374843e-01	  1.2419810e+00 	 1.9603944e-01


.. parsed-literal::

     127	 1.4926980e+00	 1.1759026e-01	 1.5471068e+00	 1.5380341e-01	  1.2417585e+00 	 2.0823550e-01


.. parsed-literal::

     128	 1.4943783e+00	 1.1747504e-01	 1.5487443e+00	 1.5381230e-01	  1.2418056e+00 	 2.0220852e-01


.. parsed-literal::

     129	 1.4971320e+00	 1.1725558e-01	 1.5514901e+00	 1.5381031e-01	  1.2393571e+00 	 2.1632195e-01


.. parsed-literal::

     130	 1.4988141e+00	 1.1691152e-01	 1.5532645e+00	 1.5361849e-01	  1.2328131e+00 	 2.1148515e-01


.. parsed-literal::

     131	 1.5009876e+00	 1.1688971e-01	 1.5554177e+00	 1.5346427e-01	  1.2329286e+00 	 2.0491648e-01
     132	 1.5027592e+00	 1.1681426e-01	 1.5572468e+00	 1.5327206e-01	  1.2299715e+00 	 1.9849539e-01


.. parsed-literal::

     133	 1.5041723e+00	 1.1676220e-01	 1.5586905e+00	 1.5312369e-01	  1.2281705e+00 	 1.7375302e-01


.. parsed-literal::

     134	 1.5073612e+00	 1.1662923e-01	 1.5619436e+00	 1.5313431e-01	  1.2212998e+00 	 2.0391536e-01


.. parsed-literal::

     135	 1.5091023e+00	 1.1655494e-01	 1.5636831e+00	 1.5299760e-01	  1.2217608e+00 	 3.2082486e-01
     136	 1.5107089e+00	 1.1649621e-01	 1.5652272e+00	 1.5310461e-01	  1.2228557e+00 	 1.8215275e-01


.. parsed-literal::

     137	 1.5123041e+00	 1.1646136e-01	 1.5667360e+00	 1.5312479e-01	  1.2249345e+00 	 1.8492985e-01


.. parsed-literal::

     138	 1.5138513e+00	 1.1656475e-01	 1.5682149e+00	 1.5319022e-01	  1.2262730e+00 	 2.0429897e-01
     139	 1.5157015e+00	 1.1659844e-01	 1.5700018e+00	 1.5301138e-01	  1.2276139e+00 	 1.9503284e-01


.. parsed-literal::

     140	 1.5178139e+00	 1.1666647e-01	 1.5721237e+00	 1.5249448e-01	  1.2266054e+00 	 1.9732213e-01
     141	 1.5191081e+00	 1.1672365e-01	 1.5734592e+00	 1.5243645e-01	  1.2239966e+00 	 1.9690180e-01


.. parsed-literal::

     142	 1.5204134e+00	 1.1668952e-01	 1.5748040e+00	 1.5232522e-01	  1.2220726e+00 	 1.9917059e-01


.. parsed-literal::

     143	 1.5229880e+00	 1.1661051e-01	 1.5775307e+00	 1.5219966e-01	  1.2155244e+00 	 2.1029997e-01
     144	 1.5240923e+00	 1.1642182e-01	 1.5787177e+00	 1.5219119e-01	  1.2103238e+00 	 1.7443538e-01


.. parsed-literal::

     145	 1.5254157e+00	 1.1641199e-01	 1.5799747e+00	 1.5224273e-01	  1.2125623e+00 	 2.0428610e-01
     146	 1.5270272e+00	 1.1637223e-01	 1.5815595e+00	 1.5237026e-01	  1.2133073e+00 	 1.7042685e-01


.. parsed-literal::

     147	 1.5280989e+00	 1.1630690e-01	 1.5826429e+00	 1.5243918e-01	  1.2136094e+00 	 2.0399308e-01


.. parsed-literal::

     148	 1.5308014e+00	 1.1612245e-01	 1.5854331e+00	 1.5262567e-01	  1.2139662e+00 	 2.0374584e-01


.. parsed-literal::

     149	 1.5320193e+00	 1.1590380e-01	 1.5867388e+00	 1.5262879e-01	  1.2111147e+00 	 2.9865265e-01
     150	 1.5335085e+00	 1.1584185e-01	 1.5883360e+00	 1.5271552e-01	  1.2080461e+00 	 1.8318415e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min, sys: 1.23 s, total: 2min 1s
    Wall time: 30.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f341c8183d0>



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
    CPU times: user 1.78 s, sys: 45 ms, total: 1.83 s
    Wall time: 569 ms


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

