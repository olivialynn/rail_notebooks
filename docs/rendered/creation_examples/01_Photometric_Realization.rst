Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f9bd90a8700>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.182480  0.181345  
    1      25.391064  0.025008  0.014416  
    2      24.304707  0.009725  0.007912  
    3      25.291103  0.123446  0.094478  
    4      25.096743  0.044881  0.024934  
    ...          ...       ...       ...  
    99995  24.737946  0.052026  0.045468  
    99996  24.224169  0.208466  0.147826  
    99997  25.613836  0.106393  0.090358  
    99998  25.274899  0.079957  0.076811  
    99999  25.699642  0.131776  0.090718  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  input: None, error_model
    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>29.816354</td>
          <td>2.477217</td>
          <td>27.168918</td>
          <td>0.243624</td>
          <td>26.005059</td>
          <td>0.079199</td>
          <td>25.212907</td>
          <td>0.064126</td>
          <td>24.599223</td>
          <td>0.071277</td>
          <td>23.902414</td>
          <td>0.086742</td>
          <td>0.182480</td>
          <td>0.181345</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.696282</td>
          <td>0.766549</td>
          <td>26.646346</td>
          <td>0.138706</td>
          <td>26.086887</td>
          <td>0.138004</td>
          <td>25.449094</td>
          <td>0.149778</td>
          <td>25.227479</td>
          <td>0.268843</td>
          <td>0.025008</td>
          <td>0.014416</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.936207</td>
          <td>1.019187</td>
          <td>28.158290</td>
          <td>0.527251</td>
          <td>27.617218</td>
          <td>0.311919</td>
          <td>25.848618</td>
          <td>0.112233</td>
          <td>24.976816</td>
          <td>0.099406</td>
          <td>24.086659</td>
          <td>0.101972</td>
          <td>0.009725</td>
          <td>0.007912</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.071740</td>
          <td>0.494786</td>
          <td>27.034458</td>
          <td>0.193138</td>
          <td>26.328084</td>
          <td>0.169697</td>
          <td>25.371972</td>
          <td>0.140165</td>
          <td>25.263520</td>
          <td>0.276843</td>
          <td>0.123446</td>
          <td>0.094478</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>28.707289</td>
          <td>1.549346</td>
          <td>26.103782</td>
          <td>0.098141</td>
          <td>25.956462</td>
          <td>0.075872</td>
          <td>25.969473</td>
          <td>0.124673</td>
          <td>25.766915</td>
          <td>0.196244</td>
          <td>25.737325</td>
          <td>0.402821</td>
          <td>0.044881</td>
          <td>0.024934</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.637748</td>
          <td>0.419213</td>
          <td>26.324392</td>
          <td>0.118968</td>
          <td>25.322267</td>
          <td>0.043234</td>
          <td>25.210753</td>
          <td>0.064004</td>
          <td>24.842183</td>
          <td>0.088321</td>
          <td>24.755404</td>
          <td>0.181543</td>
          <td>0.052026</td>
          <td>0.045468</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.283187</td>
          <td>0.669910</td>
          <td>27.020226</td>
          <td>0.215374</td>
          <td>26.191341</td>
          <td>0.093319</td>
          <td>25.219184</td>
          <td>0.064484</td>
          <td>24.848964</td>
          <td>0.088850</td>
          <td>24.297969</td>
          <td>0.122603</td>
          <td>0.208466</td>
          <td>0.147826</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>31.051485</td>
          <td>3.634224</td>
          <td>26.931975</td>
          <td>0.200043</td>
          <td>26.473958</td>
          <td>0.119470</td>
          <td>26.106414</td>
          <td>0.140347</td>
          <td>25.904452</td>
          <td>0.220188</td>
          <td>26.010743</td>
          <td>0.495110</td>
          <td>0.106393</td>
          <td>0.090358</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.304403</td>
          <td>0.679723</td>
          <td>25.967525</td>
          <td>0.087081</td>
          <td>26.117240</td>
          <td>0.087432</td>
          <td>25.773151</td>
          <td>0.105075</td>
          <td>25.903099</td>
          <td>0.219940</td>
          <td>25.151344</td>
          <td>0.252611</td>
          <td>0.079957</td>
          <td>0.076811</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.249817</td>
          <td>0.654684</td>
          <td>26.579334</td>
          <td>0.148271</td>
          <td>26.463899</td>
          <td>0.118430</td>
          <td>26.433024</td>
          <td>0.185495</td>
          <td>26.268893</td>
          <td>0.296813</td>
          <td>25.530511</td>
          <td>0.342858</td>
          <td>0.131776</td>
          <td>0.090718</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.162587</td>
          <td>0.344823</td>
          <td>26.723686</td>
          <td>0.210236</td>
          <td>25.928542</td>
          <td>0.096369</td>
          <td>25.325557</td>
          <td>0.093215</td>
          <td>24.826793</td>
          <td>0.113238</td>
          <td>23.944248</td>
          <td>0.117866</td>
          <td>0.182480</td>
          <td>0.181345</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.877982</td>
          <td>0.219333</td>
          <td>26.716334</td>
          <td>0.172624</td>
          <td>26.184393</td>
          <td>0.177004</td>
          <td>26.067066</td>
          <td>0.292855</td>
          <td>26.272824</td>
          <td>0.683787</td>
          <td>0.025008</td>
          <td>0.014416</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.605427</td>
          <td>2.245988</td>
          <td>28.006263</td>
          <td>0.485589</td>
          <td>26.121180</td>
          <td>0.167551</td>
          <td>25.027721</td>
          <td>0.122025</td>
          <td>24.423028</td>
          <td>0.160881</td>
          <td>0.009725</td>
          <td>0.007912</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.381260</td>
          <td>0.800344</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.911878</td>
          <td>0.467876</td>
          <td>26.111858</td>
          <td>0.172871</td>
          <td>25.415598</td>
          <td>0.176897</td>
          <td>24.956787</td>
          <td>0.261320</td>
          <td>0.123446</td>
          <td>0.094478</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.157609</td>
          <td>0.320906</td>
          <td>26.202813</td>
          <td>0.123799</td>
          <td>25.947575</td>
          <td>0.088944</td>
          <td>25.847626</td>
          <td>0.133055</td>
          <td>25.762539</td>
          <td>0.228912</td>
          <td>25.046368</td>
          <td>0.271926</td>
          <td>0.044881</td>
          <td>0.024934</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>28.047502</td>
          <td>1.179156</td>
          <td>26.551820</td>
          <td>0.167622</td>
          <td>25.415492</td>
          <td>0.055768</td>
          <td>24.968011</td>
          <td>0.061711</td>
          <td>24.750823</td>
          <td>0.096574</td>
          <td>24.902754</td>
          <td>0.242581</td>
          <td>0.052026</td>
          <td>0.045468</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.513214</td>
          <td>0.900898</td>
          <td>26.868944</td>
          <td>0.236774</td>
          <td>26.035877</td>
          <td>0.105636</td>
          <td>25.227818</td>
          <td>0.085349</td>
          <td>24.783877</td>
          <td>0.108847</td>
          <td>24.048962</td>
          <td>0.128792</td>
          <td>0.208466</td>
          <td>0.147826</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.324756</td>
          <td>0.767990</td>
          <td>26.505949</td>
          <td>0.164635</td>
          <td>26.367153</td>
          <td>0.131842</td>
          <td>26.331458</td>
          <td>0.206554</td>
          <td>25.539401</td>
          <td>0.195026</td>
          <td>25.711683</td>
          <td>0.469385</td>
          <td>0.106393</td>
          <td>0.090358</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.360346</td>
          <td>0.380468</td>
          <td>26.083369</td>
          <td>0.113214</td>
          <td>26.035463</td>
          <td>0.097634</td>
          <td>25.830435</td>
          <td>0.133251</td>
          <td>25.559455</td>
          <td>0.196160</td>
          <td>24.853601</td>
          <td>0.235747</td>
          <td>0.079957</td>
          <td>0.076811</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.766041</td>
          <td>0.523716</td>
          <td>27.011624</td>
          <td>0.253455</td>
          <td>26.273494</td>
          <td>0.122707</td>
          <td>26.077789</td>
          <td>0.168290</td>
          <td>25.669471</td>
          <td>0.219428</td>
          <td>26.416669</td>
          <td>0.777039</td>
          <td>0.131776</td>
          <td>0.090718</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.387800</td>
          <td>0.166154</td>
          <td>26.072949</td>
          <td>0.115389</td>
          <td>25.165380</td>
          <td>0.085617</td>
          <td>24.568182</td>
          <td>0.095335</td>
          <td>23.891036</td>
          <td>0.118879</td>
          <td>0.182480</td>
          <td>0.181345</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.888197</td>
          <td>0.507416</td>
          <td>26.891334</td>
          <td>0.194230</td>
          <td>26.803574</td>
          <td>0.159631</td>
          <td>26.354797</td>
          <td>0.174589</td>
          <td>26.085172</td>
          <td>0.256992</td>
          <td>25.512343</td>
          <td>0.339753</td>
          <td>0.025008</td>
          <td>0.014416</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.208683</td>
          <td>1.193556</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.825800</td>
          <td>0.368183</td>
          <td>25.817132</td>
          <td>0.109313</td>
          <td>25.058862</td>
          <td>0.106918</td>
          <td>24.447761</td>
          <td>0.139720</td>
          <td>0.009725</td>
          <td>0.007912</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.604909</td>
          <td>0.446288</td>
          <td>29.091281</td>
          <td>1.070359</td>
          <td>27.424874</td>
          <td>0.303786</td>
          <td>26.435698</td>
          <td>0.213792</td>
          <td>25.921243</td>
          <td>0.254669</td>
          <td>25.458947</td>
          <td>0.368878</td>
          <td>0.123446</td>
          <td>0.094478</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.923512</td>
          <td>0.524323</td>
          <td>25.879356</td>
          <td>0.081822</td>
          <td>25.751485</td>
          <td>0.064411</td>
          <td>25.572804</td>
          <td>0.089774</td>
          <td>25.622204</td>
          <td>0.176602</td>
          <td>25.353561</td>
          <td>0.302743</td>
          <td>0.044881</td>
          <td>0.024934</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.647424</td>
          <td>0.866364</td>
          <td>26.412212</td>
          <td>0.131882</td>
          <td>25.410891</td>
          <td>0.048283</td>
          <td>25.249487</td>
          <td>0.068478</td>
          <td>25.056859</td>
          <td>0.109994</td>
          <td>24.820663</td>
          <td>0.197914</td>
          <td>0.052026</td>
          <td>0.045468</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.074094</td>
          <td>0.696083</td>
          <td>26.825583</td>
          <td>0.238048</td>
          <td>26.040798</td>
          <td>0.111181</td>
          <td>25.328904</td>
          <td>0.097892</td>
          <td>24.694616</td>
          <td>0.105517</td>
          <td>24.287384</td>
          <td>0.165673</td>
          <td>0.208466</td>
          <td>0.147826</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.769370</td>
          <td>0.976115</td>
          <td>26.516543</td>
          <td>0.155308</td>
          <td>26.255250</td>
          <td>0.110857</td>
          <td>26.362761</td>
          <td>0.196495</td>
          <td>25.580643</td>
          <td>0.187558</td>
          <td>25.118496</td>
          <td>0.275107</td>
          <td>0.106393</td>
          <td>0.090358</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.218272</td>
          <td>0.317183</td>
          <td>26.364286</td>
          <td>0.131629</td>
          <td>26.072012</td>
          <td>0.090763</td>
          <td>25.941720</td>
          <td>0.131778</td>
          <td>25.488303</td>
          <td>0.166928</td>
          <td>25.134884</td>
          <td>0.268448</td>
          <td>0.079957</td>
          <td>0.076811</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.599330</td>
          <td>0.446237</td>
          <td>26.837478</td>
          <td>0.209071</td>
          <td>26.391890</td>
          <td>0.128546</td>
          <td>26.334663</td>
          <td>0.197614</td>
          <td>25.683129</td>
          <td>0.210285</td>
          <td>25.653377</td>
          <td>0.430809</td>
          <td>0.131776</td>
          <td>0.090718</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
