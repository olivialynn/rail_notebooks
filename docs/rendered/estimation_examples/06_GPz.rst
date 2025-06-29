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
       1	-3.3542719e-01	 3.1796699e-01	-3.2577044e-01	 3.3030634e-01	[-3.5025613e-01]	 4.6894240e-01


.. parsed-literal::

       2	-2.6382487e-01	 3.0710252e-01	-2.3935367e-01	 3.1848774e-01	[-2.7577567e-01]	 2.3083520e-01


.. parsed-literal::

       3	-2.1585150e-01	 2.8437797e-01	-1.7060289e-01	 2.9860889e-01	[-2.3369251e-01]	 2.8142786e-01


.. parsed-literal::

       4	-1.7695352e-01	 2.6719676e-01	-1.2675120e-01	 2.8659977e-01	[-2.2482691e-01]	 2.9888797e-01


.. parsed-literal::

       5	-1.1338794e-01	 2.5511863e-01	-8.2287103e-02	 2.7149397e-01	[-1.7122108e-01]	 2.1541286e-01


.. parsed-literal::

       6	-6.3610272e-02	 2.5067462e-01	-3.5953892e-02	 2.6535873e-01	[-9.1342901e-02]	 2.0906687e-01


.. parsed-literal::

       7	-4.5106353e-02	 2.4695956e-01	-2.1371846e-02	 2.6033553e-01	[-7.3384674e-02]	 2.1901321e-01
       8	-3.2238825e-02	 2.4486333e-01	-1.2222284e-02	 2.5707191e-01	[-6.4359303e-02]	 1.8630219e-01


.. parsed-literal::

       9	-1.9091246e-02	 2.4230921e-01	-1.2928314e-03	 2.5396390e-01	[-4.8634702e-02]	 2.1580362e-01


.. parsed-literal::

      10	-8.1031524e-03	 2.4034691e-01	 7.6746113e-03	 2.5194587e-01	[-4.3589711e-02]	 2.1886349e-01


.. parsed-literal::

      11	-2.7373242e-03	 2.3930346e-01	 1.2364539e-02	 2.5111291e-01	[-4.0166916e-02]	 2.1230793e-01


.. parsed-literal::

      12	 2.6839339e-03	 2.3845645e-01	 1.6800267e-02	 2.5027452e-01	[-3.6142904e-02]	 2.0602608e-01
      13	 8.4278712e-03	 2.3741962e-01	 2.2480694e-02	 2.4998428e-01	[-3.3060346e-02]	 1.9107437e-01


.. parsed-literal::

      14	 7.5808175e-02	 2.2406320e-01	 9.6290421e-02	 2.3740474e-01	[ 5.6651321e-02]	 4.3891764e-01


.. parsed-literal::

      15	 1.3600675e-01	 2.2034368e-01	 1.5816381e-01	 2.3314209e-01	[ 1.0791514e-01]	 2.1143603e-01


.. parsed-literal::

      16	 1.9964878e-01	 2.1524604e-01	 2.2710616e-01	 2.2934704e-01	[ 1.3123613e-01]	 2.1828294e-01


.. parsed-literal::

      17	 2.6403694e-01	 2.1543673e-01	 2.9336057e-01	 2.3128215e-01	[ 2.1254958e-01]	 2.1333647e-01


.. parsed-literal::

      18	 3.0287148e-01	 2.1356314e-01	 3.3323456e-01	 2.2777875e-01	[ 2.5687729e-01]	 2.1325254e-01
      19	 3.3443603e-01	 2.1234020e-01	 3.6481043e-01	 2.2944506e-01	[ 2.9109938e-01]	 1.7702937e-01


.. parsed-literal::

      20	 3.7920600e-01	 2.0931064e-01	 4.1161721e-01	 2.3446012e-01	[ 3.3605370e-01]	 1.8717027e-01
      21	 4.5259890e-01	 2.0904754e-01	 4.8706119e-01	 2.2555832e-01	[ 4.1929260e-01]	 1.7378783e-01


.. parsed-literal::

      22	 5.2977589e-01	 2.0219977e-01	 5.6663374e-01	 2.1665828e-01	[ 5.0027961e-01]	 2.0065355e-01


.. parsed-literal::

      23	 5.8684922e-01	 1.9712970e-01	 6.2617142e-01	 2.0849760e-01	[ 5.5846800e-01]	 2.1584487e-01
      24	 6.3755078e-01	 1.9269551e-01	 6.7733073e-01	 2.0127632e-01	[ 6.2270683e-01]	 1.8293500e-01


.. parsed-literal::

      25	 6.7708644e-01	 1.8829198e-01	 7.1718781e-01	 1.9756355e-01	[ 6.6301122e-01]	 2.0918369e-01
      26	 7.2592516e-01	 1.8576022e-01	 7.6595440e-01	 1.9657203e-01	[ 7.1546070e-01]	 1.9843030e-01


.. parsed-literal::

      27	 7.8340551e-01	 1.8784743e-01	 8.2358558e-01	 1.9836975e-01	[ 7.7602447e-01]	 2.2349119e-01


.. parsed-literal::

      28	 7.9896386e-01	 2.0694424e-01	 8.3831000e-01	 2.1812635e-01	  7.6410357e-01 	 2.0671320e-01


.. parsed-literal::

      29	 8.4474507e-01	 1.9633554e-01	 8.8482105e-01	 2.0874911e-01	[ 8.0867903e-01]	 2.0159721e-01
      30	 8.6316782e-01	 1.9142851e-01	 9.0403767e-01	 2.0305472e-01	[ 8.2976362e-01]	 2.0789957e-01


.. parsed-literal::

      31	 8.9035693e-01	 1.8998017e-01	 9.3143438e-01	 1.9964594e-01	[ 8.5110807e-01]	 1.8046045e-01


.. parsed-literal::

      32	 9.1070109e-01	 1.9128241e-01	 9.5298850e-01	 1.9820208e-01	[ 8.6386129e-01]	 2.1494365e-01


.. parsed-literal::

      33	 9.2676446e-01	 1.9001081e-01	 9.6817862e-01	 1.9706429e-01	[ 8.6989025e-01]	 2.2202635e-01


.. parsed-literal::

      34	 9.3950873e-01	 1.8732859e-01	 9.8104013e-01	 1.9383171e-01	[ 8.9066623e-01]	 2.0852041e-01


.. parsed-literal::

      35	 9.5522247e-01	 1.8440123e-01	 9.9731844e-01	 1.9079214e-01	[ 9.1266856e-01]	 2.1061730e-01


.. parsed-literal::

      36	 9.7096967e-01	 1.8387905e-01	 1.0142471e+00	 1.8964317e-01	[ 9.2739013e-01]	 2.2019982e-01


.. parsed-literal::

      37	 9.8573767e-01	 1.8175600e-01	 1.0294357e+00	 1.8760235e-01	[ 9.4345632e-01]	 2.1446896e-01


.. parsed-literal::

      38	 1.0093494e+00	 1.7867170e-01	 1.0534491e+00	 1.8502307e-01	[ 9.6626824e-01]	 2.0928931e-01


.. parsed-literal::

      39	 1.0335721e+00	 1.7409832e-01	 1.0790575e+00	 1.8175932e-01	[ 9.8712573e-01]	 2.1485758e-01
      40	 1.0515749e+00	 1.6893636e-01	 1.0978895e+00	 1.7724835e-01	[ 9.9871724e-01]	 1.8142009e-01


.. parsed-literal::

      41	 1.0640252e+00	 1.6699259e-01	 1.1101385e+00	 1.7516875e-01	[ 1.0098528e+00]	 1.8249559e-01


.. parsed-literal::

      42	 1.0855461e+00	 1.6193704e-01	 1.1321669e+00	 1.7006449e-01	[ 1.0227544e+00]	 2.0851302e-01


.. parsed-literal::

      43	 1.1028426e+00	 1.5886960e-01	 1.1493237e+00	 1.6648434e-01	[ 1.0348500e+00]	 2.0676088e-01


.. parsed-literal::

      44	 1.1165562e+00	 1.5338646e-01	 1.1642173e+00	 1.6139207e-01	[ 1.0367097e+00]	 2.1051955e-01


.. parsed-literal::

      45	 1.1326736e+00	 1.5174069e-01	 1.1799028e+00	 1.6050317e-01	[ 1.0544162e+00]	 2.1557069e-01


.. parsed-literal::

      46	 1.1435045e+00	 1.5058895e-01	 1.1905822e+00	 1.6011723e-01	[ 1.0664786e+00]	 2.1131706e-01


.. parsed-literal::

      47	 1.1560847e+00	 1.4858610e-01	 1.2032713e+00	 1.5919950e-01	[ 1.0780259e+00]	 2.0576143e-01


.. parsed-literal::

      48	 1.1777392e+00	 1.4530790e-01	 1.2254652e+00	 1.5777091e-01	[ 1.0912545e+00]	 2.1146536e-01


.. parsed-literal::

      49	 1.1905669e+00	 1.4297997e-01	 1.2383303e+00	 1.5695855e-01	[ 1.0965895e+00]	 2.1364093e-01
      50	 1.2042900e+00	 1.4159091e-01	 1.2520138e+00	 1.5562110e-01	[ 1.1082078e+00]	 2.1010828e-01


.. parsed-literal::

      51	 1.2171283e+00	 1.3972162e-01	 1.2653779e+00	 1.5397663e-01	[ 1.1136685e+00]	 2.0648384e-01


.. parsed-literal::

      52	 1.2300026e+00	 1.3815301e-01	 1.2784726e+00	 1.5288465e-01	[ 1.1200269e+00]	 2.0567369e-01


.. parsed-literal::

      53	 1.2416846e+00	 1.3611887e-01	 1.2904619e+00	 1.5232783e-01	[ 1.1326643e+00]	 3.1694460e-01


.. parsed-literal::

      54	 1.2546297e+00	 1.3509190e-01	 1.3035156e+00	 1.5243022e-01	[ 1.1375085e+00]	 2.1198249e-01


.. parsed-literal::

      55	 1.2635734e+00	 1.3419948e-01	 1.3123700e+00	 1.5220621e-01	[ 1.1472214e+00]	 2.0913291e-01


.. parsed-literal::

      56	 1.2792321e+00	 1.3234466e-01	 1.3283533e+00	 1.5198118e-01	[ 1.1608228e+00]	 2.2459269e-01


.. parsed-literal::

      57	 1.2827185e+00	 1.3161542e-01	 1.3325008e+00	 1.5184399e-01	[ 1.1646777e+00]	 2.1166897e-01
      58	 1.2942133e+00	 1.3108862e-01	 1.3436252e+00	 1.5097212e-01	[ 1.1757488e+00]	 2.0219779e-01


.. parsed-literal::

      59	 1.2993199e+00	 1.3079472e-01	 1.3488465e+00	 1.5080790e-01	[ 1.1792824e+00]	 1.8317747e-01


.. parsed-literal::

      60	 1.3080214e+00	 1.3032956e-01	 1.3578312e+00	 1.5062776e-01	[ 1.1830151e+00]	 2.1812940e-01


.. parsed-literal::

      61	 1.3208334e+00	 1.2966310e-01	 1.3709551e+00	 1.5156030e-01	[ 1.1879161e+00]	 2.1066594e-01
      62	 1.3308724e+00	 1.2921863e-01	 1.3815789e+00	 1.5363526e-01	  1.1773690e+00 	 2.0620203e-01


.. parsed-literal::

      63	 1.3433423e+00	 1.2849281e-01	 1.3935688e+00	 1.5376154e-01	[ 1.1966901e+00]	 2.0436025e-01


.. parsed-literal::

      64	 1.3497417e+00	 1.2814757e-01	 1.3999146e+00	 1.5416313e-01	[ 1.2034202e+00]	 2.1160674e-01


.. parsed-literal::

      65	 1.3585695e+00	 1.2770529e-01	 1.4089646e+00	 1.5439744e-01	[ 1.2110385e+00]	 2.1509385e-01


.. parsed-literal::

      66	 1.3640347e+00	 1.2742584e-01	 1.4147740e+00	 1.5450556e-01	[ 1.2116558e+00]	 2.1750379e-01


.. parsed-literal::

      67	 1.3727618e+00	 1.2720980e-01	 1.4234426e+00	 1.5367060e-01	[ 1.2192852e+00]	 2.0782614e-01


.. parsed-literal::

      68	 1.3791983e+00	 1.2715194e-01	 1.4300660e+00	 1.5301113e-01	[ 1.2229911e+00]	 2.0159006e-01
      69	 1.3858772e+00	 1.2701999e-01	 1.4368985e+00	 1.5194433e-01	[ 1.2262635e+00]	 1.8214941e-01


.. parsed-literal::

      70	 1.3902410e+00	 1.2649930e-01	 1.4413696e+00	 1.5147572e-01	[ 1.2370604e+00]	 2.1569276e-01


.. parsed-literal::

      71	 1.3981038e+00	 1.2606772e-01	 1.4489778e+00	 1.5082716e-01	[ 1.2398463e+00]	 2.1638370e-01
      72	 1.4020751e+00	 1.2570638e-01	 1.4529620e+00	 1.5084974e-01	  1.2391133e+00 	 1.9911480e-01


.. parsed-literal::

      73	 1.4087928e+00	 1.2497398e-01	 1.4598492e+00	 1.5112106e-01	  1.2324497e+00 	 2.0903730e-01
      74	 1.4150444e+00	 1.2451794e-01	 1.4663042e+00	 1.5115940e-01	  1.2265009e+00 	 1.7808867e-01


.. parsed-literal::

      75	 1.4223764e+00	 1.2420475e-01	 1.4737941e+00	 1.5155195e-01	  1.2198307e+00 	 2.1005607e-01


.. parsed-literal::

      76	 1.4278656e+00	 1.2391346e-01	 1.4794288e+00	 1.5110430e-01	  1.2213437e+00 	 2.1201754e-01
      77	 1.4343762e+00	 1.2349732e-01	 1.4861777e+00	 1.5075884e-01	  1.2235731e+00 	 1.7956519e-01


.. parsed-literal::

      78	 1.4375621e+00	 1.2334805e-01	 1.4896646e+00	 1.5061899e-01	  1.2208025e+00 	 1.7493796e-01


.. parsed-literal::

      79	 1.4430578e+00	 1.2318666e-01	 1.4949027e+00	 1.5072625e-01	  1.2272836e+00 	 2.1952558e-01


.. parsed-literal::

      80	 1.4467102e+00	 1.2302653e-01	 1.4985456e+00	 1.5078803e-01	  1.2251306e+00 	 2.1561337e-01


.. parsed-literal::

      81	 1.4499244e+00	 1.2282239e-01	 1.5017897e+00	 1.5046563e-01	  1.2243359e+00 	 2.0826411e-01


.. parsed-literal::

      82	 1.4537850e+00	 1.2255893e-01	 1.5058729e+00	 1.4961640e-01	  1.2143345e+00 	 2.1652150e-01


.. parsed-literal::

      83	 1.4583058e+00	 1.2240797e-01	 1.5102921e+00	 1.4909838e-01	  1.2187545e+00 	 2.0784450e-01


.. parsed-literal::

      84	 1.4608888e+00	 1.2230740e-01	 1.5128585e+00	 1.4869155e-01	  1.2214824e+00 	 2.2469711e-01


.. parsed-literal::

      85	 1.4649539e+00	 1.2215795e-01	 1.5170004e+00	 1.4816070e-01	  1.2219733e+00 	 2.1636128e-01
      86	 1.4660295e+00	 1.2179364e-01	 1.5182536e+00	 1.4797277e-01	  1.2246440e+00 	 1.9054842e-01


.. parsed-literal::

      87	 1.4704987e+00	 1.2171693e-01	 1.5226371e+00	 1.4798824e-01	  1.2253305e+00 	 2.0993948e-01


.. parsed-literal::

      88	 1.4724380e+00	 1.2157266e-01	 1.5245786e+00	 1.4818140e-01	  1.2243157e+00 	 2.1180224e-01


.. parsed-literal::

      89	 1.4751288e+00	 1.2132798e-01	 1.5273471e+00	 1.4853883e-01	  1.2221517e+00 	 2.0777273e-01


.. parsed-literal::

      90	 1.4790443e+00	 1.2107552e-01	 1.5313463e+00	 1.4917831e-01	  1.2211446e+00 	 2.1232152e-01
      91	 1.4800029e+00	 1.2075490e-01	 1.5326946e+00	 1.4998775e-01	  1.2212980e+00 	 2.0034575e-01


.. parsed-literal::

      92	 1.4873059e+00	 1.2059301e-01	 1.5397363e+00	 1.5030100e-01	  1.2284280e+00 	 2.0307612e-01


.. parsed-literal::

      93	 1.4895139e+00	 1.2047206e-01	 1.5418836e+00	 1.5001830e-01	  1.2308900e+00 	 2.1151781e-01


.. parsed-literal::

      94	 1.4926335e+00	 1.2022604e-01	 1.5450067e+00	 1.4966400e-01	  1.2332328e+00 	 2.1075106e-01
      95	 1.4949873e+00	 1.1983100e-01	 1.5474245e+00	 1.4918649e-01	  1.2297595e+00 	 1.9458365e-01


.. parsed-literal::

      96	 1.4982228e+00	 1.1966179e-01	 1.5506151e+00	 1.4907379e-01	  1.2311495e+00 	 2.1575332e-01


.. parsed-literal::

      97	 1.5016911e+00	 1.1940310e-01	 1.5541510e+00	 1.4896674e-01	  1.2294202e+00 	 2.1056962e-01


.. parsed-literal::

      98	 1.5043494e+00	 1.1923794e-01	 1.5568873e+00	 1.4873390e-01	  1.2259862e+00 	 2.1921563e-01
      99	 1.5087439e+00	 1.1889128e-01	 1.5615437e+00	 1.4807703e-01	  1.2266529e+00 	 1.9308257e-01


.. parsed-literal::

     100	 1.5122530e+00	 1.1873933e-01	 1.5651812e+00	 1.4745234e-01	  1.2153019e+00 	 2.1153474e-01
     101	 1.5143079e+00	 1.1867000e-01	 1.5670679e+00	 1.4747160e-01	  1.2233398e+00 	 1.8232465e-01


.. parsed-literal::

     102	 1.5166322e+00	 1.1843310e-01	 1.5693846e+00	 1.4740717e-01	  1.2287299e+00 	 1.8540907e-01
     103	 1.5192182e+00	 1.1805615e-01	 1.5720675e+00	 1.4734285e-01	  1.2299501e+00 	 1.7446351e-01


.. parsed-literal::

     104	 1.5220951e+00	 1.1767438e-01	 1.5749952e+00	 1.4740311e-01	  1.2324861e+00 	 2.1150947e-01


.. parsed-literal::

     105	 1.5249101e+00	 1.1742487e-01	 1.5778471e+00	 1.4750263e-01	  1.2314224e+00 	 2.0788097e-01
     106	 1.5281527e+00	 1.1703749e-01	 1.5812800e+00	 1.4794487e-01	  1.2256066e+00 	 1.7353225e-01


.. parsed-literal::

     107	 1.5307719e+00	 1.1701274e-01	 1.5839652e+00	 1.4807490e-01	  1.2249627e+00 	 2.0968008e-01


.. parsed-literal::

     108	 1.5325261e+00	 1.1698816e-01	 1.5856831e+00	 1.4814572e-01	  1.2273293e+00 	 2.1243978e-01
     109	 1.5363202e+00	 1.1666736e-01	 1.5895249e+00	 1.4852649e-01	  1.2340090e+00 	 1.8657446e-01


.. parsed-literal::

     110	 1.5376765e+00	 1.1647717e-01	 1.5908705e+00	 1.4879464e-01	  1.2321940e+00 	 2.0497251e-01


.. parsed-literal::

     111	 1.5395416e+00	 1.1640828e-01	 1.5926959e+00	 1.4878047e-01	  1.2339797e+00 	 2.0513630e-01


.. parsed-literal::

     112	 1.5411686e+00	 1.1626257e-01	 1.5943244e+00	 1.4880050e-01	  1.2338827e+00 	 2.1123862e-01
     113	 1.5428439e+00	 1.1613587e-01	 1.5960321e+00	 1.4877859e-01	  1.2321534e+00 	 1.8908143e-01


.. parsed-literal::

     114	 1.5463737e+00	 1.1585550e-01	 1.5996496e+00	 1.4862837e-01	  1.2249990e+00 	 2.1150374e-01


.. parsed-literal::

     115	 1.5479661e+00	 1.1567726e-01	 1.6013554e+00	 1.4844561e-01	  1.2185291e+00 	 3.3088636e-01
     116	 1.5510121e+00	 1.1537441e-01	 1.6044240e+00	 1.4821036e-01	  1.2116211e+00 	 1.9981790e-01


.. parsed-literal::

     117	 1.5536563e+00	 1.1506703e-01	 1.6071176e+00	 1.4794613e-01	  1.2032944e+00 	 1.9784141e-01
     118	 1.5559153e+00	 1.1471746e-01	 1.6094083e+00	 1.4762908e-01	  1.2001920e+00 	 1.8245029e-01


.. parsed-literal::

     119	 1.5579327e+00	 1.1460603e-01	 1.6114462e+00	 1.4760801e-01	  1.1968593e+00 	 2.0227551e-01


.. parsed-literal::

     120	 1.5594642e+00	 1.1455091e-01	 1.6129738e+00	 1.4762729e-01	  1.1954423e+00 	 2.1600056e-01


.. parsed-literal::

     121	 1.5609123e+00	 1.1451096e-01	 1.6144667e+00	 1.4762526e-01	  1.1929327e+00 	 2.0501304e-01
     122	 1.5629909e+00	 1.1444417e-01	 1.6166645e+00	 1.4756284e-01	  1.1811972e+00 	 1.9831252e-01


.. parsed-literal::

     123	 1.5648736e+00	 1.1437586e-01	 1.6186527e+00	 1.4763222e-01	  1.1716190e+00 	 2.1227074e-01


.. parsed-literal::

     124	 1.5665423e+00	 1.1438979e-01	 1.6203169e+00	 1.4760551e-01	  1.1697519e+00 	 2.0721436e-01
     125	 1.5677638e+00	 1.1428323e-01	 1.6215415e+00	 1.4754178e-01	  1.1641100e+00 	 1.8589067e-01


.. parsed-literal::

     126	 1.5694839e+00	 1.1411837e-01	 1.6232663e+00	 1.4749847e-01	  1.1586335e+00 	 1.9981241e-01
     127	 1.5721071e+00	 1.1389104e-01	 1.6260089e+00	 1.4731649e-01	  1.1498392e+00 	 2.0010662e-01


.. parsed-literal::

     128	 1.5739960e+00	 1.1369958e-01	 1.6279218e+00	 1.4725393e-01	  1.1405827e+00 	 2.1243453e-01


.. parsed-literal::

     129	 1.5753774e+00	 1.1372400e-01	 1.6292498e+00	 1.4711306e-01	  1.1484680e+00 	 2.1143341e-01


.. parsed-literal::

     130	 1.5766768e+00	 1.1369088e-01	 1.6305671e+00	 1.4701854e-01	  1.1498900e+00 	 2.2043037e-01


.. parsed-literal::

     131	 1.5780441e+00	 1.1360756e-01	 1.6320077e+00	 1.4681982e-01	  1.1491243e+00 	 2.1399760e-01


.. parsed-literal::

     132	 1.5798416e+00	 1.1339048e-01	 1.6338520e+00	 1.4672632e-01	  1.1440833e+00 	 2.1080208e-01


.. parsed-literal::

     133	 1.5815095e+00	 1.1314631e-01	 1.6355801e+00	 1.4670429e-01	  1.1314853e+00 	 2.1055770e-01


.. parsed-literal::

     134	 1.5829422e+00	 1.1287538e-01	 1.6370452e+00	 1.4682617e-01	  1.1242242e+00 	 2.1607399e-01
     135	 1.5840530e+00	 1.1282782e-01	 1.6381149e+00	 1.4686736e-01	  1.1244268e+00 	 1.7871547e-01


.. parsed-literal::

     136	 1.5859129e+00	 1.1269284e-01	 1.6399677e+00	 1.4687693e-01	  1.1211803e+00 	 2.1250224e-01
     137	 1.5869875e+00	 1.1255620e-01	 1.6410994e+00	 1.4676362e-01	  1.1150696e+00 	 1.8148136e-01


.. parsed-literal::

     138	 1.5882860e+00	 1.1254068e-01	 1.6423637e+00	 1.4667599e-01	  1.1146938e+00 	 1.8776059e-01


.. parsed-literal::

     139	 1.5893131e+00	 1.1250419e-01	 1.6434002e+00	 1.4652664e-01	  1.1124335e+00 	 2.2103763e-01
     140	 1.5901637e+00	 1.1247108e-01	 1.6442579e+00	 1.4639176e-01	  1.1117408e+00 	 2.1031904e-01


.. parsed-literal::

     141	 1.5922432e+00	 1.1235426e-01	 1.6463642e+00	 1.4622236e-01	  1.1061511e+00 	 1.9872856e-01


.. parsed-literal::

     142	 1.5934770e+00	 1.1227309e-01	 1.6476582e+00	 1.4588465e-01	  1.1091589e+00 	 2.1124673e-01
     143	 1.5949946e+00	 1.1221159e-01	 1.6491096e+00	 1.4603104e-01	  1.1102984e+00 	 1.9053054e-01


.. parsed-literal::

     144	 1.5958896e+00	 1.1215221e-01	 1.6499969e+00	 1.4616258e-01	  1.1115734e+00 	 1.8468952e-01


.. parsed-literal::

     145	 1.5971563e+00	 1.1205585e-01	 1.6513000e+00	 1.4627052e-01	  1.1128382e+00 	 2.0987582e-01
     146	 1.5985241e+00	 1.1188811e-01	 1.6527888e+00	 1.4651478e-01	  1.1140964e+00 	 1.7998600e-01


.. parsed-literal::

     147	 1.5998203e+00	 1.1187401e-01	 1.6540882e+00	 1.4652046e-01	  1.1117577e+00 	 1.9046521e-01


.. parsed-literal::

     148	 1.6008149e+00	 1.1186319e-01	 1.6551028e+00	 1.4649957e-01	  1.1097490e+00 	 2.0368147e-01


.. parsed-literal::

     149	 1.6022006e+00	 1.1184870e-01	 1.6565344e+00	 1.4663626e-01	  1.1035519e+00 	 2.2260523e-01


.. parsed-literal::

     150	 1.6030799e+00	 1.1183414e-01	 1.6574851e+00	 1.4678393e-01	  1.1012474e+00 	 3.0697441e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.01 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fe669bdbbb0>



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
    CPU times: user 1.89 s, sys: 42 ms, total: 1.93 s
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

