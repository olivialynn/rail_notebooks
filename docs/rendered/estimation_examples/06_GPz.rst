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
       1	-3.3985783e-01	 3.1926898e-01	-3.3010873e-01	 3.2526304e-01	[-3.4180406e-01]	 4.6573734e-01


.. parsed-literal::

       2	-2.6602185e-01	 3.0742228e-01	-2.4085341e-01	 3.1548429e-01	[-2.6577034e-01]	 2.3333120e-01


.. parsed-literal::

       3	-2.2278526e-01	 2.8802693e-01	-1.8119689e-01	 2.9685506e-01	[-2.2144017e-01]	 2.9353571e-01


.. parsed-literal::

       4	-1.8705112e-01	 2.6246478e-01	-1.4444914e-01	 2.7557292e-01	 -2.2900285e-01 	 2.0471168e-01


.. parsed-literal::

       5	-9.7563285e-02	 2.5430933e-01	-6.0915929e-02	 2.6981670e-01	[-1.4075962e-01]	 2.1254277e-01
       6	-5.8982456e-02	 2.4805646e-01	-2.6005296e-02	 2.6417768e-01	[-8.8984003e-02]	 1.9552708e-01


.. parsed-literal::

       7	-4.1550207e-02	 2.4565384e-01	-1.5557963e-02	 2.6255150e-01	[-8.5603382e-02]	 2.1328998e-01
       8	-2.7533982e-02	 2.4340269e-01	-6.0881241e-03	 2.6049557e-01	[-7.8254270e-02]	 1.7898512e-01


.. parsed-literal::

       9	-1.3247489e-02	 2.4088394e-01	 4.7583834e-03	 2.5852601e-01	[-7.1289446e-02]	 2.0669317e-01


.. parsed-literal::

      10	-2.7066248e-03	 2.3915123e-01	 1.2771406e-02	 2.5601168e-01	[-5.9581152e-02]	 2.2040153e-01


.. parsed-literal::

      11	 2.5788640e-03	 2.3825828e-01	 1.6842417e-02	 2.5561106e-01	 -6.0547281e-02 	 2.1038604e-01
      12	 5.0711397e-03	 2.3775154e-01	 1.9154440e-02	 2.5519044e-01	[-5.5958272e-02]	 1.7844558e-01


.. parsed-literal::

      13	 8.3842386e-03	 2.3715577e-01	 2.2209303e-02	 2.5463224e-01	[-5.3569228e-02]	 2.2057939e-01


.. parsed-literal::

      14	 1.2795868e-02	 2.3621855e-01	 2.6981778e-02	 2.5346270e-01	[-4.5125962e-02]	 2.0225167e-01


.. parsed-literal::

      15	 5.6533779e-02	 2.2455680e-01	 7.5296196e-02	 2.3657256e-01	[ 4.7608384e-02]	 3.1906414e-01
      16	 1.5199112e-01	 2.1991015e-01	 1.7351004e-01	 2.3123412e-01	[ 1.3735228e-01]	 1.8672943e-01


.. parsed-literal::

      17	 2.3521533e-01	 2.1459545e-01	 2.6456096e-01	 2.2740117e-01	[ 1.9244076e-01]	 2.1657777e-01


.. parsed-literal::

      18	 2.8576241e-01	 2.1650685e-01	 3.1745283e-01	 2.2954763e-01	[ 2.6207221e-01]	 2.1331763e-01
      19	 3.1727591e-01	 2.1233786e-01	 3.4855600e-01	 2.2671525e-01	[ 2.9313193e-01]	 1.8319440e-01


.. parsed-literal::

      20	 3.5209188e-01	 2.0841492e-01	 3.8281270e-01	 2.2219523e-01	[ 3.3296890e-01]	 2.1147609e-01


.. parsed-literal::

      21	 4.0938669e-01	 2.0514583e-01	 4.4159496e-01	 2.1700877e-01	[ 3.9746075e-01]	 2.1998501e-01


.. parsed-literal::

      22	 5.0528479e-01	 2.0401899e-01	 5.4023084e-01	 2.1459573e-01	[ 4.9306962e-01]	 2.1344090e-01


.. parsed-literal::

      23	 5.6082732e-01	 2.0454416e-01	 5.9854071e-01	 2.1408895e-01	[ 5.6748766e-01]	 2.0365572e-01


.. parsed-literal::

      24	 5.9636225e-01	 2.0163360e-01	 6.3369213e-01	 2.1201070e-01	[ 6.0697527e-01]	 2.1681523e-01


.. parsed-literal::

      25	 6.2941748e-01	 1.9980896e-01	 6.6738319e-01	 2.1150111e-01	[ 6.4453151e-01]	 2.1919918e-01


.. parsed-literal::

      26	 6.7571917e-01	 1.9836061e-01	 7.1343909e-01	 2.1083451e-01	[ 6.9057324e-01]	 2.2453022e-01


.. parsed-literal::

      27	 7.1683613e-01	 2.0399764e-01	 7.5254657e-01	 2.1474068e-01	[ 7.2599150e-01]	 2.1423864e-01


.. parsed-literal::

      28	 7.4808998e-01	 1.9668171e-01	 7.8407665e-01	 2.0701375e-01	[ 7.6234256e-01]	 2.1442461e-01


.. parsed-literal::

      29	 7.7551808e-01	 1.9423548e-01	 8.1390866e-01	 2.0427176e-01	[ 7.8838429e-01]	 2.2041130e-01


.. parsed-literal::

      30	 8.0436588e-01	 1.9477157e-01	 8.4369298e-01	 2.0552978e-01	[ 8.0662750e-01]	 2.0429969e-01
      31	 8.1119044e-01	 1.9151857e-01	 8.5230316e-01	 2.0259627e-01	[ 8.1370802e-01]	 1.8650651e-01


.. parsed-literal::

      32	 8.4762456e-01	 1.8823266e-01	 8.8749281e-01	 1.9954128e-01	[ 8.5591620e-01]	 2.0780373e-01


.. parsed-literal::

      33	 8.7023898e-01	 1.8784089e-01	 9.1014813e-01	 1.9945384e-01	[ 8.7373228e-01]	 2.0309496e-01


.. parsed-literal::

      34	 8.9567224e-01	 1.8629798e-01	 9.3614262e-01	 1.9792198e-01	[ 8.9873867e-01]	 2.1190286e-01
      35	 9.2600073e-01	 1.8291401e-01	 9.6854248e-01	 1.9461233e-01	[ 9.2629681e-01]	 1.7730427e-01


.. parsed-literal::

      36	 9.3413405e-01	 1.8271042e-01	 9.7758407e-01	 1.9413685e-01	[ 9.2653937e-01]	 2.0723987e-01
      37	 9.5636061e-01	 1.8033713e-01	 9.9947025e-01	 1.9257641e-01	[ 9.4735500e-01]	 1.9803905e-01


.. parsed-literal::

      38	 9.7242373e-01	 1.7821502e-01	 1.0159593e+00	 1.9099557e-01	[ 9.6107735e-01]	 1.9206715e-01


.. parsed-literal::

      39	 9.8467003e-01	 1.7732566e-01	 1.0289326e+00	 1.9038553e-01	[ 9.7062047e-01]	 2.0551252e-01
      40	 9.9855711e-01	 1.7678225e-01	 1.0447911e+00	 1.9038518e-01	[ 9.7682949e-01]	 1.7928100e-01


.. parsed-literal::

      41	 1.0139709e+00	 1.7679783e-01	 1.0602957e+00	 1.9128704e-01	[ 9.8789912e-01]	 2.1096611e-01
      42	 1.0222770e+00	 1.7518496e-01	 1.0677463e+00	 1.8934017e-01	[ 1.0014472e+00]	 2.0102763e-01


.. parsed-literal::

      43	 1.0337150e+00	 1.7293384e-01	 1.0791400e+00	 1.8731757e-01	[ 1.0131428e+00]	 2.0737529e-01


.. parsed-literal::

      44	 1.0462889e+00	 1.7023972e-01	 1.0921236e+00	 1.8511155e-01	[ 1.0276362e+00]	 2.0685267e-01


.. parsed-literal::

      45	 1.0566144e+00	 1.6901056e-01	 1.1028252e+00	 1.8506568e-01	[ 1.0421661e+00]	 2.1275401e-01


.. parsed-literal::

      46	 1.0695320e+00	 1.6794799e-01	 1.1157184e+00	 1.8377734e-01	[ 1.0556388e+00]	 2.2159934e-01


.. parsed-literal::

      47	 1.0762989e+00	 1.6731646e-01	 1.1224584e+00	 1.8308307e-01	[ 1.0612113e+00]	 2.2003746e-01
      48	 1.0879358e+00	 1.6548558e-01	 1.1344578e+00	 1.8189403e-01	[ 1.0676192e+00]	 1.8478322e-01


.. parsed-literal::

      49	 1.1011009e+00	 1.6201938e-01	 1.1477478e+00	 1.7977552e-01	[ 1.0838528e+00]	 2.0991516e-01


.. parsed-literal::

      50	 1.1113507e+00	 1.5847926e-01	 1.1586081e+00	 1.7762700e-01	[ 1.0867781e+00]	 2.1957612e-01


.. parsed-literal::

      51	 1.1211458e+00	 1.5718915e-01	 1.1684074e+00	 1.7639149e-01	[ 1.0989688e+00]	 2.1144962e-01


.. parsed-literal::

      52	 1.1342825e+00	 1.5567728e-01	 1.1818968e+00	 1.7433118e-01	[ 1.1130835e+00]	 2.2122049e-01


.. parsed-literal::

      53	 1.1456413e+00	 1.5497864e-01	 1.1937395e+00	 1.7348946e-01	[ 1.1235837e+00]	 2.1050882e-01


.. parsed-literal::

      54	 1.1542960e+00	 1.5465703e-01	 1.2029714e+00	 1.7242048e-01	[ 1.1333552e+00]	 2.0811033e-01


.. parsed-literal::

      55	 1.1605221e+00	 1.5407807e-01	 1.2093808e+00	 1.7186348e-01	[ 1.1363992e+00]	 2.0833397e-01
      56	 1.1725313e+00	 1.5302504e-01	 1.2220693e+00	 1.7089965e-01	[ 1.1456369e+00]	 1.9322753e-01


.. parsed-literal::

      57	 1.1825543e+00	 1.5204859e-01	 1.2321460e+00	 1.7045470e-01	[ 1.1601050e+00]	 2.1284842e-01


.. parsed-literal::

      58	 1.1938819e+00	 1.5133726e-01	 1.2438108e+00	 1.6976704e-01	[ 1.1761564e+00]	 2.0823336e-01


.. parsed-literal::

      59	 1.2051970e+00	 1.5122208e-01	 1.2554522e+00	 1.6938738e-01	[ 1.1840477e+00]	 2.0598435e-01


.. parsed-literal::

      60	 1.2169701e+00	 1.5064842e-01	 1.2672867e+00	 1.6838122e-01	[ 1.1935127e+00]	 2.1219420e-01


.. parsed-literal::

      61	 1.2257298e+00	 1.5094450e-01	 1.2761695e+00	 1.6815784e-01	[ 1.1966424e+00]	 2.0614648e-01


.. parsed-literal::

      62	 1.2331700e+00	 1.5101978e-01	 1.2838990e+00	 1.6815765e-01	[ 1.1976556e+00]	 2.1871138e-01


.. parsed-literal::

      63	 1.2402765e+00	 1.5026653e-01	 1.2911373e+00	 1.6740901e-01	[ 1.2008736e+00]	 2.0396829e-01


.. parsed-literal::

      64	 1.2477501e+00	 1.4943520e-01	 1.2992367e+00	 1.6694162e-01	  1.2006727e+00 	 2.1214890e-01


.. parsed-literal::

      65	 1.2577712e+00	 1.4848135e-01	 1.3090832e+00	 1.6615604e-01	[ 1.2121245e+00]	 2.1162748e-01
      66	 1.2662636e+00	 1.4749611e-01	 1.3177010e+00	 1.6557070e-01	[ 1.2186369e+00]	 1.9603562e-01


.. parsed-literal::

      67	 1.2726541e+00	 1.4663949e-01	 1.3245898e+00	 1.6582011e-01	  1.2142968e+00 	 2.0757461e-01


.. parsed-literal::

      68	 1.2783828e+00	 1.4664916e-01	 1.3300796e+00	 1.6567337e-01	[ 1.2250525e+00]	 2.1502924e-01
      69	 1.2825219e+00	 1.4659108e-01	 1.3342333e+00	 1.6558043e-01	[ 1.2259871e+00]	 1.8335819e-01


.. parsed-literal::

      70	 1.2905478e+00	 1.4600104e-01	 1.3427508e+00	 1.6528104e-01	  1.2227733e+00 	 2.1717095e-01


.. parsed-literal::

      71	 1.2971126e+00	 1.4500063e-01	 1.3494201e+00	 1.6456325e-01	[ 1.2279130e+00]	 2.1329713e-01


.. parsed-literal::

      72	 1.3042801e+00	 1.4362431e-01	 1.3566364e+00	 1.6342773e-01	[ 1.2364999e+00]	 2.1602488e-01
      73	 1.3097431e+00	 1.4218000e-01	 1.3622970e+00	 1.6207853e-01	  1.2361713e+00 	 2.0434785e-01


.. parsed-literal::

      74	 1.3151679e+00	 1.4155250e-01	 1.3676902e+00	 1.6132459e-01	[ 1.2406675e+00]	 1.9762874e-01


.. parsed-literal::

      75	 1.3225837e+00	 1.4070480e-01	 1.3753190e+00	 1.6021284e-01	[ 1.2430339e+00]	 2.1041942e-01


.. parsed-literal::

      76	 1.3254447e+00	 1.4048808e-01	 1.3784703e+00	 1.5956894e-01	  1.2378052e+00 	 2.0935726e-01


.. parsed-literal::

      77	 1.3302530e+00	 1.4024345e-01	 1.3829860e+00	 1.5942027e-01	[ 1.2473736e+00]	 2.2465825e-01


.. parsed-literal::

      78	 1.3343437e+00	 1.3991517e-01	 1.3869705e+00	 1.5912896e-01	[ 1.2542124e+00]	 2.2278738e-01


.. parsed-literal::

      79	 1.3384435e+00	 1.3952750e-01	 1.3910205e+00	 1.5873305e-01	[ 1.2606627e+00]	 2.1052647e-01


.. parsed-literal::

      80	 1.3440917e+00	 1.3893994e-01	 1.3967433e+00	 1.5825838e-01	[ 1.2681118e+00]	 2.1276999e-01


.. parsed-literal::

      81	 1.3496564e+00	 1.3804281e-01	 1.4023142e+00	 1.5739593e-01	[ 1.2755428e+00]	 2.2095370e-01


.. parsed-literal::

      82	 1.3529761e+00	 1.3781753e-01	 1.4055847e+00	 1.5724571e-01	[ 1.2763894e+00]	 2.1142840e-01
      83	 1.3588311e+00	 1.3737692e-01	 1.4115566e+00	 1.5677314e-01	[ 1.2772836e+00]	 1.9992161e-01


.. parsed-literal::

      84	 1.3623138e+00	 1.3670362e-01	 1.4151789e+00	 1.5631179e-01	  1.2727171e+00 	 2.0858479e-01


.. parsed-literal::

      85	 1.3676789e+00	 1.3647527e-01	 1.4204161e+00	 1.5594420e-01	[ 1.2792705e+00]	 2.1246862e-01


.. parsed-literal::

      86	 1.3712822e+00	 1.3619353e-01	 1.4239939e+00	 1.5557960e-01	[ 1.2832253e+00]	 2.0780325e-01


.. parsed-literal::

      87	 1.3742429e+00	 1.3594622e-01	 1.4270169e+00	 1.5524388e-01	[ 1.2838435e+00]	 2.1376157e-01


.. parsed-literal::

      88	 1.3773909e+00	 1.3526136e-01	 1.4305416e+00	 1.5466577e-01	  1.2800113e+00 	 2.0848608e-01


.. parsed-literal::

      89	 1.3826694e+00	 1.3503840e-01	 1.4357966e+00	 1.5433847e-01	  1.2812076e+00 	 2.0813608e-01


.. parsed-literal::

      90	 1.3850897e+00	 1.3485432e-01	 1.4382270e+00	 1.5423060e-01	  1.2825462e+00 	 2.0411468e-01


.. parsed-literal::

      91	 1.3887776e+00	 1.3444066e-01	 1.4420776e+00	 1.5397543e-01	  1.2824208e+00 	 2.0786166e-01


.. parsed-literal::

      92	 1.3925153e+00	 1.3408864e-01	 1.4459323e+00	 1.5370086e-01	[ 1.2852052e+00]	 2.1676898e-01
      93	 1.3963401e+00	 1.3363961e-01	 1.4498031e+00	 1.5347885e-01	[ 1.2860027e+00]	 1.9473195e-01


.. parsed-literal::

      94	 1.4000373e+00	 1.3332104e-01	 1.4534970e+00	 1.5332911e-01	[ 1.2869504e+00]	 2.1852255e-01


.. parsed-literal::

      95	 1.4041557e+00	 1.3302562e-01	 1.4577191e+00	 1.5303443e-01	  1.2866749e+00 	 2.0416284e-01


.. parsed-literal::

      96	 1.4061801e+00	 1.3231897e-01	 1.4600156e+00	 1.5277193e-01	  1.2749354e+00 	 2.1275520e-01


.. parsed-literal::

      97	 1.4098443e+00	 1.3239547e-01	 1.4635522e+00	 1.5267159e-01	  1.2825269e+00 	 2.0250821e-01


.. parsed-literal::

      98	 1.4123272e+00	 1.3228712e-01	 1.4660773e+00	 1.5254128e-01	  1.2844050e+00 	 2.1719432e-01
      99	 1.4162400e+00	 1.3197835e-01	 1.4700352e+00	 1.5233065e-01	  1.2861295e+00 	 1.9120669e-01


.. parsed-literal::

     100	 1.4200905e+00	 1.3152007e-01	 1.4739015e+00	 1.5203851e-01	  1.2848287e+00 	 1.7675591e-01
     101	 1.4240753e+00	 1.3121990e-01	 1.4777754e+00	 1.5188923e-01	[ 1.2899829e+00]	 1.9756651e-01


.. parsed-literal::

     102	 1.4260074e+00	 1.3109005e-01	 1.4796562e+00	 1.5180271e-01	[ 1.2908998e+00]	 1.8747258e-01
     103	 1.4288955e+00	 1.3080195e-01	 1.4825912e+00	 1.5160780e-01	  1.2891539e+00 	 1.8801332e-01


.. parsed-literal::

     104	 1.4313185e+00	 1.3049694e-01	 1.4851404e+00	 1.5119331e-01	  1.2873261e+00 	 2.1898246e-01


.. parsed-literal::

     105	 1.4339809e+00	 1.3033482e-01	 1.4878292e+00	 1.5100320e-01	  1.2888145e+00 	 2.1311760e-01


.. parsed-literal::

     106	 1.4367422e+00	 1.3005627e-01	 1.4906064e+00	 1.5064614e-01	  1.2891735e+00 	 2.1839643e-01
     107	 1.4389641e+00	 1.2987108e-01	 1.4928213e+00	 1.5036418e-01	  1.2907853e+00 	 1.9982433e-01


.. parsed-literal::

     108	 1.4418350e+00	 1.2969453e-01	 1.4956403e+00	 1.4999338e-01	[ 1.2922313e+00]	 2.0857716e-01
     109	 1.4445019e+00	 1.2927905e-01	 1.4982831e+00	 1.4968593e-01	  1.2900172e+00 	 1.7693830e-01


.. parsed-literal::

     110	 1.4468172e+00	 1.2913414e-01	 1.5005998e+00	 1.4953501e-01	  1.2908933e+00 	 1.9703197e-01


.. parsed-literal::

     111	 1.4492283e+00	 1.2890975e-01	 1.5031271e+00	 1.4951798e-01	  1.2881345e+00 	 2.1841860e-01


.. parsed-literal::

     112	 1.4527080e+00	 1.2847277e-01	 1.5069499e+00	 1.4937280e-01	  1.2815061e+00 	 2.0330238e-01


.. parsed-literal::

     113	 1.4546789e+00	 1.2841503e-01	 1.5091269e+00	 1.4945129e-01	  1.2773664e+00 	 2.1024990e-01


.. parsed-literal::

     114	 1.4572720e+00	 1.2830666e-01	 1.5116542e+00	 1.4936454e-01	  1.2793388e+00 	 2.1793079e-01


.. parsed-literal::

     115	 1.4594803e+00	 1.2822784e-01	 1.5138284e+00	 1.4930058e-01	  1.2794995e+00 	 2.1192837e-01


.. parsed-literal::

     116	 1.4614291e+00	 1.2821961e-01	 1.5157526e+00	 1.4933455e-01	  1.2793898e+00 	 2.1304750e-01
     117	 1.4649998e+00	 1.2821097e-01	 1.5193500e+00	 1.4929936e-01	  1.2799450e+00 	 1.9901037e-01


.. parsed-literal::

     118	 1.4672699e+00	 1.2825708e-01	 1.5219113e+00	 1.4965207e-01	  1.2648635e+00 	 2.1014524e-01


.. parsed-literal::

     119	 1.4712848e+00	 1.2807981e-01	 1.5258900e+00	 1.4935776e-01	  1.2703982e+00 	 2.0576525e-01


.. parsed-literal::

     120	 1.4729724e+00	 1.2799071e-01	 1.5275651e+00	 1.4921798e-01	  1.2720143e+00 	 2.0886397e-01
     121	 1.4743585e+00	 1.2788537e-01	 1.5290218e+00	 1.4920521e-01	  1.2684895e+00 	 1.9773984e-01


.. parsed-literal::

     122	 1.4761793e+00	 1.2784244e-01	 1.5309468e+00	 1.4926188e-01	  1.2609558e+00 	 1.6780233e-01
     123	 1.4787119e+00	 1.2770264e-01	 1.5335204e+00	 1.4935060e-01	  1.2524282e+00 	 2.0135307e-01


.. parsed-literal::

     124	 1.4808950e+00	 1.2753015e-01	 1.5355584e+00	 1.4946289e-01	  1.2512444e+00 	 2.1427560e-01


.. parsed-literal::

     125	 1.4823663e+00	 1.2742156e-01	 1.5368867e+00	 1.4944577e-01	  1.2549144e+00 	 2.0480800e-01


.. parsed-literal::

     126	 1.4841369e+00	 1.2731194e-01	 1.5385165e+00	 1.4943462e-01	  1.2589987e+00 	 2.1514916e-01
     127	 1.4861470e+00	 1.2717906e-01	 1.5405222e+00	 1.4960502e-01	  1.2497394e+00 	 1.8544197e-01


.. parsed-literal::

     128	 1.4880290e+00	 1.2710360e-01	 1.5424450e+00	 1.4961891e-01	  1.2483147e+00 	 2.1287131e-01
     129	 1.4897066e+00	 1.2712511e-01	 1.5442952e+00	 1.4978523e-01	  1.2404920e+00 	 1.8518138e-01


.. parsed-literal::

     130	 1.4907573e+00	 1.2708056e-01	 1.5455028e+00	 1.4985677e-01	  1.2292246e+00 	 2.0437670e-01


.. parsed-literal::

     131	 1.4920170e+00	 1.2714682e-01	 1.5468189e+00	 1.4997089e-01	  1.2250359e+00 	 2.0670557e-01
     132	 1.4939493e+00	 1.2746299e-01	 1.5488496e+00	 1.5034382e-01	  1.2163000e+00 	 1.9944930e-01


.. parsed-literal::

     133	 1.4949129e+00	 1.2749971e-01	 1.5498422e+00	 1.5034833e-01	  1.2149387e+00 	 2.1383595e-01
     134	 1.4960552e+00	 1.2754457e-01	 1.5509433e+00	 1.5032944e-01	  1.2167324e+00 	 1.8817544e-01


.. parsed-literal::

     135	 1.4976879e+00	 1.2758157e-01	 1.5525601e+00	 1.5018978e-01	  1.2190516e+00 	 2.0772099e-01


.. parsed-literal::

     136	 1.4986343e+00	 1.2755243e-01	 1.5535324e+00	 1.5013433e-01	  1.2178906e+00 	 2.1912503e-01


.. parsed-literal::

     137	 1.5005228e+00	 1.2742729e-01	 1.5556545e+00	 1.4989479e-01	  1.2054932e+00 	 2.1072173e-01


.. parsed-literal::

     138	 1.5023890e+00	 1.2724668e-01	 1.5575916e+00	 1.4991922e-01	  1.1956334e+00 	 2.1436930e-01
     139	 1.5033750e+00	 1.2711834e-01	 1.5585632e+00	 1.4989275e-01	  1.1927939e+00 	 1.8308353e-01


.. parsed-literal::

     140	 1.5051334e+00	 1.2679396e-01	 1.5604204e+00	 1.4978136e-01	  1.1786916e+00 	 2.1178460e-01


.. parsed-literal::

     141	 1.5066352e+00	 1.2657321e-01	 1.5619929e+00	 1.4977435e-01	  1.1722124e+00 	 2.0811391e-01


.. parsed-literal::

     142	 1.5084449e+00	 1.2624163e-01	 1.5639040e+00	 1.4973024e-01	  1.1500564e+00 	 2.1555400e-01


.. parsed-literal::

     143	 1.5099212e+00	 1.2631402e-01	 1.5653600e+00	 1.4986684e-01	  1.1516841e+00 	 2.1061087e-01


.. parsed-literal::

     144	 1.5113527e+00	 1.2640961e-01	 1.5667140e+00	 1.5007274e-01	  1.1568112e+00 	 2.1208692e-01


.. parsed-literal::

     145	 1.5127935e+00	 1.2628162e-01	 1.5681284e+00	 1.5010292e-01	  1.1537368e+00 	 2.1967816e-01


.. parsed-literal::

     146	 1.5146565e+00	 1.2607420e-01	 1.5700243e+00	 1.5026464e-01	  1.1476282e+00 	 2.1857882e-01


.. parsed-literal::

     147	 1.5162296e+00	 1.2571852e-01	 1.5717044e+00	 1.5023708e-01	  1.1340053e+00 	 2.1533251e-01
     148	 1.5177844e+00	 1.2534991e-01	 1.5733931e+00	 1.5008404e-01	  1.1173367e+00 	 1.8799019e-01


.. parsed-literal::

     149	 1.5190338e+00	 1.2527954e-01	 1.5746802e+00	 1.4994969e-01	  1.1180969e+00 	 2.1579170e-01


.. parsed-literal::

     150	 1.5205887e+00	 1.2507784e-01	 1.5763412e+00	 1.4985944e-01	  1.1138251e+00 	 2.0971012e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.16 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fad14902050>



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
    CPU times: user 2.12 s, sys: 43 ms, total: 2.16 s
    Wall time: 658 ms


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

