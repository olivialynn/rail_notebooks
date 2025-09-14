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
       1	-3.4168756e-01	 3.2021497e-01	-3.3196031e-01	 3.2229370e-01	[-3.3517354e-01]	 4.8328948e-01


.. parsed-literal::

       2	-2.7101074e-01	 3.0926766e-01	-2.4664236e-01	 3.1184825e-01	[-2.5447553e-01]	 2.4397039e-01


.. parsed-literal::

       3	-2.2890095e-01	 2.8995241e-01	-1.8750799e-01	 2.9434447e-01	[-2.0588529e-01]	 3.1468177e-01
       4	-1.9192926e-01	 2.6574877e-01	-1.4885873e-01	 2.7199402e-01	[-1.8285607e-01]	 1.9717050e-01


.. parsed-literal::

       5	-1.0910200e-01	 2.5722390e-01	-7.4298070e-02	 2.6249663e-01	[-1.0258064e-01]	 2.2141647e-01


.. parsed-literal::

       6	-6.8086797e-02	 2.5069297e-01	-3.6750761e-02	 2.5709253e-01	[-6.1368884e-02]	 2.2556210e-01


.. parsed-literal::

       7	-5.2176679e-02	 2.4842468e-01	-2.6662620e-02	 2.5474212e-01	[-5.2464051e-02]	 2.1930528e-01
       8	-3.4866705e-02	 2.4540885e-01	-1.4082806e-02	 2.5091726e-01	[-3.7018906e-02]	 1.8474531e-01


.. parsed-literal::

       9	-2.0941032e-02	 2.4284767e-01	-3.4274660e-03	 2.4780313e-01	[-2.5288735e-02]	 2.2436833e-01


.. parsed-literal::

      10	-1.1947835e-02	 2.4138199e-01	 3.0399917e-03	 2.4653370e-01	[-1.7752579e-02]	 2.0444107e-01


.. parsed-literal::

      11	-4.8519306e-03	 2.4004487e-01	 9.6070336e-03	 2.4566851e-01	[-1.4886464e-02]	 2.1595931e-01
      12	-2.0520993e-03	 2.3956859e-01	 1.2029841e-02	 2.4537627e-01	[-1.3299183e-02]	 1.7733169e-01


.. parsed-literal::

      13	 2.0047524e-03	 2.3880165e-01	 1.6093492e-02	 2.4479370e-01	[-1.0528892e-02]	 2.0742393e-01


.. parsed-literal::

      14	 8.0609130e-02	 2.2532452e-01	 9.8797174e-02	 2.3232917e-01	[ 7.4874668e-02]	 3.4150696e-01


.. parsed-literal::

      15	 1.3286408e-01	 2.2254212e-01	 1.5401415e-01	 2.3165019e-01	[ 1.2876905e-01]	 2.1419907e-01


.. parsed-literal::

      16	 2.4065149e-01	 2.2042714e-01	 2.6618519e-01	 2.2938616e-01	[ 2.3059416e-01]	 2.1830297e-01


.. parsed-literal::

      17	 2.9746199e-01	 2.1682670e-01	 3.2754603e-01	 2.2540777e-01	[ 2.8372145e-01]	 2.2466135e-01


.. parsed-literal::

      18	 3.6982851e-01	 2.1084292e-01	 4.0237186e-01	 2.2186581e-01	[ 3.4447144e-01]	 2.2152543e-01


.. parsed-literal::

      19	 4.3357468e-01	 2.1016144e-01	 4.6765050e-01	 2.2258515e-01	[ 4.0384961e-01]	 2.2731638e-01


.. parsed-literal::

      20	 4.9486600e-01	 2.0660435e-01	 5.2917143e-01	 2.1589439e-01	[ 4.7804780e-01]	 2.1894431e-01


.. parsed-literal::

      21	 5.6440412e-01	 2.0141518e-01	 6.0035098e-01	 2.1046597e-01	[ 5.4385748e-01]	 2.2613668e-01
      22	 6.2402927e-01	 1.9913796e-01	 6.6494974e-01	 2.0824258e-01	[ 5.8565169e-01]	 1.8922997e-01


.. parsed-literal::

      23	 6.7037382e-01	 1.9358244e-01	 7.0947955e-01	 2.0189324e-01	[ 6.5630657e-01]	 1.8601704e-01


.. parsed-literal::

      24	 7.0052462e-01	 1.8993336e-01	 7.3837657e-01	 1.9813182e-01	[ 6.9161040e-01]	 2.1481442e-01


.. parsed-literal::

      25	 7.2532559e-01	 1.8904758e-01	 7.6254842e-01	 1.9659288e-01	[ 7.2094944e-01]	 3.3227587e-01


.. parsed-literal::

      26	 7.4904418e-01	 1.8831865e-01	 7.8572699e-01	 1.9469380e-01	[ 7.4894168e-01]	 2.0804095e-01
      27	 7.7969084e-01	 1.8715588e-01	 8.1707923e-01	 1.9377244e-01	[ 7.7311343e-01]	 1.8810511e-01


.. parsed-literal::

      28	 8.0077193e-01	 1.8710919e-01	 8.3912387e-01	 1.9279432e-01	[ 7.9230563e-01]	 2.0332360e-01


.. parsed-literal::

      29	 8.2280962e-01	 1.8747793e-01	 8.6174242e-01	 1.9263670e-01	[ 8.1287049e-01]	 2.1416402e-01


.. parsed-literal::

      30	 8.4485281e-01	 1.8476335e-01	 8.8383632e-01	 1.8997559e-01	[ 8.4001356e-01]	 2.1460462e-01
      31	 8.6077488e-01	 1.8248876e-01	 8.9976143e-01	 1.8857853e-01	[ 8.5130240e-01]	 1.9836497e-01


.. parsed-literal::

      32	 8.8636507e-01	 1.7838916e-01	 9.2643082e-01	 1.8640316e-01	[ 8.7147244e-01]	 2.1161866e-01
      33	 9.0201767e-01	 1.7687269e-01	 9.4245198e-01	 1.8469323e-01	[ 8.8106414e-01]	 1.8037820e-01


.. parsed-literal::

      34	 9.1275972e-01	 1.7575128e-01	 9.5334187e-01	 1.8403099e-01	[ 8.9910917e-01]	 2.1231508e-01


.. parsed-literal::

      35	 9.2482358e-01	 1.7432126e-01	 9.6582360e-01	 1.8325978e-01	[ 9.1635495e-01]	 2.0558834e-01
      36	 9.3752077e-01	 1.7420402e-01	 9.7875577e-01	 1.8223587e-01	[ 9.3090336e-01]	 1.7740369e-01


.. parsed-literal::

      37	 9.6126108e-01	 1.7427934e-01	 1.0036220e+00	 1.8159581e-01	[ 9.6115571e-01]	 2.0739007e-01


.. parsed-literal::

      38	 9.8064726e-01	 1.7523437e-01	 1.0231974e+00	 1.8050722e-01	[ 9.8229579e-01]	 2.1564174e-01


.. parsed-literal::

      39	 9.9227851e-01	 1.7499517e-01	 1.0348993e+00	 1.7995135e-01	[ 9.9573470e-01]	 2.1310949e-01


.. parsed-literal::

      40	 1.0129052e+00	 1.7514629e-01	 1.0564653e+00	 1.7922300e-01	[ 1.0181568e+00]	 2.2068548e-01
      41	 1.0279102e+00	 1.7376736e-01	 1.0724137e+00	 1.7652332e-01	[ 1.0420117e+00]	 1.8852401e-01


.. parsed-literal::

      42	 1.0383070e+00	 1.7367180e-01	 1.0835624e+00	 1.7494340e-01	[ 1.0546957e+00]	 2.0743847e-01


.. parsed-literal::

      43	 1.0458061e+00	 1.7242362e-01	 1.0910057e+00	 1.7380629e-01	[ 1.0616865e+00]	 2.0627761e-01
      44	 1.0557999e+00	 1.7136386e-01	 1.1012968e+00	 1.7279712e-01	[ 1.0699246e+00]	 1.8673635e-01


.. parsed-literal::

      45	 1.0673105e+00	 1.6968478e-01	 1.1129282e+00	 1.7132218e-01	[ 1.0764265e+00]	 2.1276236e-01


.. parsed-literal::

      46	 1.0785669e+00	 1.6956024e-01	 1.1251381e+00	 1.7094281e-01	[ 1.0813235e+00]	 2.1500444e-01


.. parsed-literal::

      47	 1.0901573e+00	 1.6777663e-01	 1.1362753e+00	 1.6999374e-01	[ 1.0899521e+00]	 2.1456718e-01


.. parsed-literal::

      48	 1.0965710e+00	 1.6715380e-01	 1.1426090e+00	 1.6897759e-01	[ 1.0974422e+00]	 2.1725011e-01


.. parsed-literal::

      49	 1.1071615e+00	 1.6662025e-01	 1.1535385e+00	 1.6756008e-01	[ 1.1114548e+00]	 2.2332621e-01


.. parsed-literal::

      50	 1.1149620e+00	 1.6538310e-01	 1.1617443e+00	 1.6488139e-01	[ 1.1248358e+00]	 2.2315431e-01


.. parsed-literal::

      51	 1.1229122e+00	 1.6425574e-01	 1.1698567e+00	 1.6409874e-01	[ 1.1327119e+00]	 2.1037602e-01


.. parsed-literal::

      52	 1.1294996e+00	 1.6284909e-01	 1.1766866e+00	 1.6299131e-01	[ 1.1380406e+00]	 2.2151613e-01


.. parsed-literal::

      53	 1.1359464e+00	 1.6153465e-01	 1.1835288e+00	 1.6151745e-01	[ 1.1442643e+00]	 2.1647048e-01
      54	 1.1469272e+00	 1.5857356e-01	 1.1950907e+00	 1.5837761e-01	[ 1.1547963e+00]	 1.8678045e-01


.. parsed-literal::

      55	 1.1567403e+00	 1.5688018e-01	 1.2050940e+00	 1.5714443e-01	[ 1.1671186e+00]	 2.0944238e-01


.. parsed-literal::

      56	 1.1637925e+00	 1.5636989e-01	 1.2120051e+00	 1.5695249e-01	[ 1.1717694e+00]	 2.0923710e-01


.. parsed-literal::

      57	 1.1740199e+00	 1.5576489e-01	 1.2225062e+00	 1.5669832e-01	[ 1.1751295e+00]	 2.1345329e-01


.. parsed-literal::

      58	 1.1807613e+00	 1.5452313e-01	 1.2293217e+00	 1.5631117e-01	[ 1.1778372e+00]	 2.2082663e-01


.. parsed-literal::

      59	 1.1880365e+00	 1.5400664e-01	 1.2366404e+00	 1.5595972e-01	[ 1.1839265e+00]	 2.1820545e-01


.. parsed-literal::

      60	 1.1965254e+00	 1.5291631e-01	 1.2454850e+00	 1.5495579e-01	[ 1.1921748e+00]	 2.0136166e-01


.. parsed-literal::

      61	 1.2040986e+00	 1.5246803e-01	 1.2533105e+00	 1.5428531e-01	[ 1.1994935e+00]	 2.0725536e-01


.. parsed-literal::

      62	 1.2093458e+00	 1.5116656e-01	 1.2594611e+00	 1.5311607e-01	[ 1.2006338e+00]	 2.0235896e-01


.. parsed-literal::

      63	 1.2167181e+00	 1.5184666e-01	 1.2664183e+00	 1.5354483e-01	[ 1.2134617e+00]	 2.1094894e-01


.. parsed-literal::

      64	 1.2214790e+00	 1.5133117e-01	 1.2712212e+00	 1.5273074e-01	[ 1.2195700e+00]	 2.1034741e-01


.. parsed-literal::

      65	 1.2264325e+00	 1.5098569e-01	 1.2763380e+00	 1.5226486e-01	[ 1.2254495e+00]	 2.0439529e-01


.. parsed-literal::

      66	 1.2339918e+00	 1.5052614e-01	 1.2844952e+00	 1.5140523e-01	[ 1.2381378e+00]	 2.2063899e-01
      67	 1.2389859e+00	 1.5053313e-01	 1.2894782e+00	 1.5092821e-01	[ 1.2458070e+00]	 2.0153999e-01


.. parsed-literal::

      68	 1.2464214e+00	 1.4999181e-01	 1.2968047e+00	 1.5070191e-01	[ 1.2546408e+00]	 2.0299411e-01


.. parsed-literal::

      69	 1.2529672e+00	 1.4922088e-01	 1.3033120e+00	 1.5031493e-01	[ 1.2607693e+00]	 2.1674895e-01


.. parsed-literal::

      70	 1.2580931e+00	 1.4861125e-01	 1.3083468e+00	 1.4994016e-01	[ 1.2636910e+00]	 2.1225548e-01


.. parsed-literal::

      71	 1.2662955e+00	 1.4693229e-01	 1.3167812e+00	 1.4907577e-01	[ 1.2665407e+00]	 2.0482254e-01
      72	 1.2713876e+00	 1.4665966e-01	 1.3218913e+00	 1.4795150e-01	[ 1.2726783e+00]	 1.6846967e-01


.. parsed-literal::

      73	 1.2769814e+00	 1.4625155e-01	 1.3273716e+00	 1.4790138e-01	[ 1.2771345e+00]	 2.1297383e-01


.. parsed-literal::

      74	 1.2804935e+00	 1.4596452e-01	 1.3309967e+00	 1.4751890e-01	[ 1.2822751e+00]	 2.1095157e-01
      75	 1.2865648e+00	 1.4543386e-01	 1.3372414e+00	 1.4705114e-01	[ 1.2884710e+00]	 2.0465970e-01


.. parsed-literal::

      76	 1.2938414e+00	 1.4424487e-01	 1.3448400e+00	 1.4583957e-01	[ 1.2966703e+00]	 2.0415187e-01


.. parsed-literal::

      77	 1.3011755e+00	 1.4362012e-01	 1.3521809e+00	 1.4573249e-01	[ 1.3026777e+00]	 2.0900869e-01


.. parsed-literal::

      78	 1.3051677e+00	 1.4344277e-01	 1.3560261e+00	 1.4583096e-01	[ 1.3039710e+00]	 2.1114969e-01
      79	 1.3105476e+00	 1.4291048e-01	 1.3616407e+00	 1.4605212e-01	  1.3024746e+00 	 1.7404294e-01


.. parsed-literal::

      80	 1.3150450e+00	 1.4258869e-01	 1.3663320e+00	 1.4604957e-01	  1.3020063e+00 	 2.0978808e-01


.. parsed-literal::

      81	 1.3185294e+00	 1.4247667e-01	 1.3699085e+00	 1.4596369e-01	[ 1.3047706e+00]	 2.1514893e-01
      82	 1.3237938e+00	 1.4197120e-01	 1.3754337e+00	 1.4560243e-01	[ 1.3073022e+00]	 1.8689728e-01


.. parsed-literal::

      83	 1.3275605e+00	 1.4153719e-01	 1.3793176e+00	 1.4539969e-01	[ 1.3073418e+00]	 2.0645952e-01


.. parsed-literal::

      84	 1.3323934e+00	 1.4107492e-01	 1.3841018e+00	 1.4506799e-01	[ 1.3094486e+00]	 2.2337556e-01


.. parsed-literal::

      85	 1.3377927e+00	 1.4027115e-01	 1.3896253e+00	 1.4465168e-01	  1.3084213e+00 	 2.1626043e-01
      86	 1.3412780e+00	 1.3998841e-01	 1.3931699e+00	 1.4456839e-01	[ 1.3105027e+00]	 1.9272494e-01


.. parsed-literal::

      87	 1.3469160e+00	 1.3941808e-01	 1.3989262e+00	 1.4443311e-01	[ 1.3132155e+00]	 2.0827627e-01


.. parsed-literal::

      88	 1.3507830e+00	 1.3922051e-01	 1.4030799e+00	 1.4464006e-01	[ 1.3165186e+00]	 2.0558619e-01


.. parsed-literal::

      89	 1.3553664e+00	 1.3914481e-01	 1.4074758e+00	 1.4454481e-01	[ 1.3217059e+00]	 2.1108150e-01
      90	 1.3591531e+00	 1.3900601e-01	 1.4112112e+00	 1.4455030e-01	[ 1.3238416e+00]	 1.8985963e-01


.. parsed-literal::

      91	 1.3630034e+00	 1.3886932e-01	 1.4150911e+00	 1.4467451e-01	[ 1.3250289e+00]	 1.8812656e-01


.. parsed-literal::

      92	 1.3672688e+00	 1.3849372e-01	 1.4196125e+00	 1.4504199e-01	  1.3181422e+00 	 2.1324015e-01
      93	 1.3726909e+00	 1.3821515e-01	 1.4250727e+00	 1.4537803e-01	  1.3220741e+00 	 1.7656732e-01


.. parsed-literal::

      94	 1.3758956e+00	 1.3792989e-01	 1.4282726e+00	 1.4513334e-01	[ 1.3263982e+00]	 1.9199300e-01


.. parsed-literal::

      95	 1.3790349e+00	 1.3744996e-01	 1.4315448e+00	 1.4505435e-01	[ 1.3267797e+00]	 2.2628999e-01


.. parsed-literal::

      96	 1.3835969e+00	 1.3700122e-01	 1.4363312e+00	 1.4511139e-01	  1.3263241e+00 	 2.1057463e-01


.. parsed-literal::

      97	 1.3884992e+00	 1.3672012e-01	 1.4413058e+00	 1.4542092e-01	[ 1.3290107e+00]	 2.1160007e-01


.. parsed-literal::

      98	 1.3920989e+00	 1.3656506e-01	 1.4448583e+00	 1.4551042e-01	[ 1.3339568e+00]	 2.1675849e-01


.. parsed-literal::

      99	 1.3960216e+00	 1.3664067e-01	 1.4487736e+00	 1.4585793e-01	[ 1.3411152e+00]	 2.1360946e-01


.. parsed-literal::

     100	 1.4008247e+00	 1.3658794e-01	 1.4537662e+00	 1.4612666e-01	[ 1.3488829e+00]	 2.0949960e-01


.. parsed-literal::

     101	 1.4053603e+00	 1.3676252e-01	 1.4582468e+00	 1.4643019e-01	[ 1.3561773e+00]	 2.0903301e-01


.. parsed-literal::

     102	 1.4094327e+00	 1.3683331e-01	 1.4623092e+00	 1.4650816e-01	[ 1.3592985e+00]	 2.2036362e-01
     103	 1.4128518e+00	 1.3674137e-01	 1.4657427e+00	 1.4629436e-01	[ 1.3619894e+00]	 1.9517255e-01


.. parsed-literal::

     104	 1.4157077e+00	 1.3673478e-01	 1.4686799e+00	 1.4612841e-01	  1.3578375e+00 	 2.1904421e-01


.. parsed-literal::

     105	 1.4201762e+00	 1.3643894e-01	 1.4730329e+00	 1.4579598e-01	[ 1.3646875e+00]	 2.2023010e-01


.. parsed-literal::

     106	 1.4222284e+00	 1.3632463e-01	 1.4750962e+00	 1.4581020e-01	[ 1.3646969e+00]	 2.1804595e-01


.. parsed-literal::

     107	 1.4249459e+00	 1.3621990e-01	 1.4778815e+00	 1.4581300e-01	  1.3625317e+00 	 2.0777488e-01
     108	 1.4286546e+00	 1.3627422e-01	 1.4818234e+00	 1.4641790e-01	  1.3557037e+00 	 1.9233727e-01


.. parsed-literal::

     109	 1.4324217e+00	 1.3631684e-01	 1.4856782e+00	 1.4660708e-01	  1.3538257e+00 	 2.2090125e-01
     110	 1.4351202e+00	 1.3634768e-01	 1.4884039e+00	 1.4679154e-01	  1.3549617e+00 	 1.9949436e-01


.. parsed-literal::

     111	 1.4379588e+00	 1.3647591e-01	 1.4913254e+00	 1.4718345e-01	  1.3579960e+00 	 2.3071265e-01


.. parsed-literal::

     112	 1.4413461e+00	 1.3634572e-01	 1.4948893e+00	 1.4701038e-01	  1.3627813e+00 	 2.1504354e-01


.. parsed-literal::

     113	 1.4442406e+00	 1.3644776e-01	 1.4978663e+00	 1.4747045e-01	  1.3630769e+00 	 2.0375466e-01


.. parsed-literal::

     114	 1.4459278e+00	 1.3624757e-01	 1.4995235e+00	 1.4719793e-01	[ 1.3673515e+00]	 2.1454096e-01
     115	 1.4479225e+00	 1.3603828e-01	 1.5015505e+00	 1.4699246e-01	[ 1.3681100e+00]	 1.9308496e-01


.. parsed-literal::

     116	 1.4515113e+00	 1.3568648e-01	 1.5052651e+00	 1.4669968e-01	  1.3669478e+00 	 2.2184014e-01


.. parsed-literal::

     117	 1.4535641e+00	 1.3548587e-01	 1.5074429e+00	 1.4651526e-01	[ 1.3688266e+00]	 2.3860550e-01


.. parsed-literal::

     118	 1.4563605e+00	 1.3541431e-01	 1.5101016e+00	 1.4641894e-01	[ 1.3690062e+00]	 2.3044205e-01


.. parsed-literal::

     119	 1.4584472e+00	 1.3528074e-01	 1.5121186e+00	 1.4627032e-01	  1.3688150e+00 	 2.2255015e-01
     120	 1.4604369e+00	 1.3507477e-01	 1.5140680e+00	 1.4605123e-01	[ 1.3691258e+00]	 1.9681048e-01


.. parsed-literal::

     121	 1.4639523e+00	 1.3459763e-01	 1.5176447e+00	 1.4558825e-01	[ 1.3696574e+00]	 2.1088433e-01


.. parsed-literal::

     122	 1.4663421e+00	 1.3437392e-01	 1.5200838e+00	 1.4546968e-01	  1.3678339e+00 	 2.1240187e-01


.. parsed-literal::

     123	 1.4686518e+00	 1.3424892e-01	 1.5223538e+00	 1.4529153e-01	[ 1.3738918e+00]	 2.1791649e-01


.. parsed-literal::

     124	 1.4702784e+00	 1.3412931e-01	 1.5240326e+00	 1.4517258e-01	[ 1.3761154e+00]	 2.2388029e-01


.. parsed-literal::

     125	 1.4724893e+00	 1.3395221e-01	 1.5263917e+00	 1.4520298e-01	[ 1.3782094e+00]	 2.1941257e-01


.. parsed-literal::

     126	 1.4746714e+00	 1.3376143e-01	 1.5287817e+00	 1.4500283e-01	[ 1.3814693e+00]	 2.0781136e-01


.. parsed-literal::

     127	 1.4766658e+00	 1.3372969e-01	 1.5307280e+00	 1.4517556e-01	  1.3805244e+00 	 2.0584941e-01


.. parsed-literal::

     128	 1.4780655e+00	 1.3374628e-01	 1.5320877e+00	 1.4535169e-01	  1.3802135e+00 	 2.1523762e-01


.. parsed-literal::

     129	 1.4796806e+00	 1.3369869e-01	 1.5337307e+00	 1.4544693e-01	  1.3803247e+00 	 2.1342874e-01


.. parsed-literal::

     130	 1.4822422e+00	 1.3364066e-01	 1.5364509e+00	 1.4590087e-01	[ 1.3829202e+00]	 2.1310830e-01


.. parsed-literal::

     131	 1.4844514e+00	 1.3343599e-01	 1.5387514e+00	 1.4578885e-01	  1.3812192e+00 	 2.0602345e-01


.. parsed-literal::

     132	 1.4858170e+00	 1.3335834e-01	 1.5400575e+00	 1.4569672e-01	[ 1.3829622e+00]	 2.2307539e-01


.. parsed-literal::

     133	 1.4875013e+00	 1.3324840e-01	 1.5417719e+00	 1.4581756e-01	[ 1.3850367e+00]	 2.1517801e-01


.. parsed-literal::

     134	 1.4890053e+00	 1.3317219e-01	 1.5433462e+00	 1.4585695e-01	  1.3829648e+00 	 2.1767688e-01


.. parsed-literal::

     135	 1.4908583e+00	 1.3314576e-01	 1.5451998e+00	 1.4599050e-01	  1.3833439e+00 	 2.0611143e-01


.. parsed-literal::

     136	 1.4932623e+00	 1.3311990e-01	 1.5476862e+00	 1.4638413e-01	  1.3807919e+00 	 2.1310639e-01


.. parsed-literal::

     137	 1.4949814e+00	 1.3311044e-01	 1.5493503e+00	 1.4613640e-01	  1.3805486e+00 	 2.1874428e-01


.. parsed-literal::

     138	 1.4961928e+00	 1.3309164e-01	 1.5505080e+00	 1.4598755e-01	  1.3805961e+00 	 2.1973634e-01


.. parsed-literal::

     139	 1.4980087e+00	 1.3311574e-01	 1.5523462e+00	 1.4591484e-01	  1.3777647e+00 	 2.1180606e-01


.. parsed-literal::

     140	 1.4995102e+00	 1.3310528e-01	 1.5538914e+00	 1.4586357e-01	  1.3742595e+00 	 2.1405029e-01


.. parsed-literal::

     141	 1.5018817e+00	 1.3297253e-01	 1.5563931e+00	 1.4575534e-01	  1.3713234e+00 	 2.1245980e-01


.. parsed-literal::

     142	 1.5037000e+00	 1.3279384e-01	 1.5582968e+00	 1.4537985e-01	  1.3700103e+00 	 2.0950890e-01


.. parsed-literal::

     143	 1.5054313e+00	 1.3268697e-01	 1.5600137e+00	 1.4527503e-01	  1.3727156e+00 	 2.2324753e-01
     144	 1.5067956e+00	 1.3252798e-01	 1.5613717e+00	 1.4515687e-01	  1.3771818e+00 	 2.0168829e-01


.. parsed-literal::

     145	 1.5082205e+00	 1.3251907e-01	 1.5628492e+00	 1.4530447e-01	  1.3779387e+00 	 2.0646548e-01


.. parsed-literal::

     146	 1.5099214e+00	 1.3242262e-01	 1.5646034e+00	 1.4539761e-01	  1.3771780e+00 	 2.2528791e-01
     147	 1.5116072e+00	 1.3245494e-01	 1.5663894e+00	 1.4553920e-01	  1.3764891e+00 	 2.0228577e-01


.. parsed-literal::

     148	 1.5132895e+00	 1.3261167e-01	 1.5681383e+00	 1.4595289e-01	  1.3739602e+00 	 1.8284321e-01


.. parsed-literal::

     149	 1.5144725e+00	 1.3263477e-01	 1.5693144e+00	 1.4599970e-01	  1.3738741e+00 	 2.2113013e-01
     150	 1.5157899e+00	 1.3268116e-01	 1.5706785e+00	 1.4614728e-01	  1.3733680e+00 	 1.9375038e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.17 s, total: 2min 8s
    Wall time: 32.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fc66c302da0>



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
    CPU times: user 2.27 s, sys: 47.9 ms, total: 2.32 s
    Wall time: 770 ms


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

