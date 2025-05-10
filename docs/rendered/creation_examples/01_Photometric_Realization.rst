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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fcd78593a90>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>26.321945</td>
          <td>0.327833</td>
          <td>26.549415</td>
          <td>0.144509</td>
          <td>25.924091</td>
          <td>0.073732</td>
          <td>25.172384</td>
          <td>0.061863</td>
          <td>24.769366</td>
          <td>0.082835</td>
          <td>24.013929</td>
          <td>0.095675</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.684214</td>
          <td>0.872891</td>
          <td>27.806272</td>
          <td>0.405032</td>
          <td>26.770440</td>
          <td>0.154321</td>
          <td>26.326994</td>
          <td>0.169539</td>
          <td>25.718795</td>
          <td>0.188445</td>
          <td>25.603390</td>
          <td>0.363067</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.111403</td>
          <td>1.585766</td>
          <td>26.076451</td>
          <td>0.136767</td>
          <td>24.993732</td>
          <td>0.100890</td>
          <td>24.372307</td>
          <td>0.130763</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.313705</td>
          <td>0.243759</td>
          <td>26.036246</td>
          <td>0.132097</td>
          <td>25.607598</td>
          <td>0.171501</td>
          <td>25.796538</td>
          <td>0.421512</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.769456</td>
          <td>0.208933</td>
          <td>26.170602</td>
          <td>0.104048</td>
          <td>25.949096</td>
          <td>0.075380</td>
          <td>25.607529</td>
          <td>0.090871</td>
          <td>25.645863</td>
          <td>0.177166</td>
          <td>24.984007</td>
          <td>0.219969</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>26.813078</td>
          <td>0.478427</td>
          <td>26.292973</td>
          <td>0.115762</td>
          <td>25.427948</td>
          <td>0.047485</td>
          <td>25.178422</td>
          <td>0.062195</td>
          <td>24.789469</td>
          <td>0.084315</td>
          <td>24.924779</td>
          <td>0.209360</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.950663</td>
          <td>0.203204</td>
          <td>26.016072</td>
          <td>0.079973</td>
          <td>25.257628</td>
          <td>0.066719</td>
          <td>24.899975</td>
          <td>0.092925</td>
          <td>24.080387</td>
          <td>0.101414</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.191802</td>
          <td>0.295455</td>
          <td>26.672795</td>
          <td>0.160621</td>
          <td>26.365367</td>
          <td>0.108684</td>
          <td>26.051319</td>
          <td>0.133830</td>
          <td>25.667368</td>
          <td>0.180426</td>
          <td>25.848302</td>
          <td>0.438429</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.513019</td>
          <td>0.380857</td>
          <td>26.271715</td>
          <td>0.113641</td>
          <td>26.095552</td>
          <td>0.085778</td>
          <td>25.838251</td>
          <td>0.111223</td>
          <td>25.422566</td>
          <td>0.146404</td>
          <td>25.181903</td>
          <td>0.259019</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.194118</td>
          <td>0.629836</td>
          <td>26.800292</td>
          <td>0.179017</td>
          <td>26.792579</td>
          <td>0.157274</td>
          <td>26.147346</td>
          <td>0.145381</td>
          <td>26.456941</td>
          <td>0.344806</td>
          <td>25.424604</td>
          <td>0.315205</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.568583</td>
          <td>0.168840</td>
          <td>26.239068</td>
          <td>0.114293</td>
          <td>25.206161</td>
          <td>0.075553</td>
          <td>24.755696</td>
          <td>0.096213</td>
          <td>23.867768</td>
          <td>0.099443</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.697947</td>
          <td>0.422265</td>
          <td>26.756743</td>
          <td>0.178444</td>
          <td>26.127596</td>
          <td>0.168464</td>
          <td>26.261030</td>
          <td>0.341528</td>
          <td>25.418078</td>
          <td>0.364436</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.477014</td>
          <td>0.843205</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.729963</td>
          <td>0.401588</td>
          <td>26.139520</td>
          <td>0.173987</td>
          <td>25.085681</td>
          <td>0.131132</td>
          <td>24.618258</td>
          <td>0.194052</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.582483</td>
          <td>0.923848</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.225459</td>
          <td>0.280546</td>
          <td>26.369572</td>
          <td>0.220695</td>
          <td>25.531455</td>
          <td>0.200337</td>
          <td>25.547259</td>
          <td>0.427638</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.078444</td>
          <td>0.300378</td>
          <td>26.282224</td>
          <td>0.132120</td>
          <td>25.935724</td>
          <td>0.087659</td>
          <td>25.548020</td>
          <td>0.102093</td>
          <td>25.636838</td>
          <td>0.205338</td>
          <td>24.902153</td>
          <td>0.240664</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>26.292534</td>
          <td>0.361026</td>
          <td>26.542562</td>
          <td>0.168206</td>
          <td>25.598350</td>
          <td>0.066446</td>
          <td>25.036496</td>
          <td>0.066466</td>
          <td>24.770801</td>
          <td>0.099560</td>
          <td>24.735399</td>
          <td>0.213804</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.641492</td>
          <td>0.925971</td>
          <td>26.625394</td>
          <td>0.177828</td>
          <td>26.008641</td>
          <td>0.093822</td>
          <td>25.116288</td>
          <td>0.070087</td>
          <td>24.701135</td>
          <td>0.092097</td>
          <td>24.214515</td>
          <td>0.135034</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.343128</td>
          <td>0.373467</td>
          <td>26.742214</td>
          <td>0.197721</td>
          <td>26.467959</td>
          <td>0.141129</td>
          <td>26.515443</td>
          <td>0.236230</td>
          <td>25.982540</td>
          <td>0.276373</td>
          <td>25.080654</td>
          <td>0.281829</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.072496</td>
          <td>0.647242</td>
          <td>26.211585</td>
          <td>0.127784</td>
          <td>26.003573</td>
          <td>0.095985</td>
          <td>25.613296</td>
          <td>0.111598</td>
          <td>25.679769</td>
          <td>0.219214</td>
          <td>24.939972</td>
          <td>0.255795</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.595185</td>
          <td>0.452094</td>
          <td>26.521026</td>
          <td>0.163570</td>
          <td>26.352580</td>
          <td>0.127396</td>
          <td>26.222462</td>
          <td>0.184388</td>
          <td>26.104583</td>
          <td>0.304226</td>
          <td>26.629885</td>
          <td>0.870809</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>27.679000</td>
          <td>0.870071</td>
          <td>26.617199</td>
          <td>0.153180</td>
          <td>26.038287</td>
          <td>0.081567</td>
          <td>25.160574</td>
          <td>0.061227</td>
          <td>24.623512</td>
          <td>0.072834</td>
          <td>23.959985</td>
          <td>0.091260</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.442825</td>
          <td>2.148958</td>
          <td>30.206695</td>
          <td>1.774477</td>
          <td>26.707540</td>
          <td>0.146346</td>
          <td>26.280696</td>
          <td>0.163137</td>
          <td>26.167839</td>
          <td>0.273743</td>
          <td>25.194841</td>
          <td>0.262014</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.752903</td>
          <td>2.479980</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.995020</td>
          <td>0.449866</td>
          <td>26.469134</td>
          <td>0.207531</td>
          <td>24.993336</td>
          <td>0.109410</td>
          <td>23.979880</td>
          <td>0.101061</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.001663</td>
          <td>0.555287</td>
          <td>26.984732</td>
          <td>0.229515</td>
          <td>26.964455</td>
          <td>0.356078</td>
          <td>26.064762</td>
          <td>0.309398</td>
          <td>25.809992</td>
          <td>0.518701</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.866660</td>
          <td>0.498214</td>
          <td>26.064334</td>
          <td>0.094923</td>
          <td>26.054327</td>
          <td>0.082836</td>
          <td>25.541965</td>
          <td>0.085904</td>
          <td>25.524550</td>
          <td>0.159999</td>
          <td>25.019193</td>
          <td>0.226815</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>26.841261</td>
          <td>0.511973</td>
          <td>26.244843</td>
          <td>0.118869</td>
          <td>25.457633</td>
          <td>0.052801</td>
          <td>25.098772</td>
          <td>0.062987</td>
          <td>24.966871</td>
          <td>0.106579</td>
          <td>24.848992</td>
          <td>0.212362</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.475142</td>
          <td>0.373588</td>
          <td>26.943855</td>
          <td>0.204828</td>
          <td>25.958235</td>
          <td>0.077262</td>
          <td>25.264620</td>
          <td>0.068315</td>
          <td>24.727694</td>
          <td>0.081177</td>
          <td>24.181521</td>
          <td>0.112685</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.950942</td>
          <td>1.052308</td>
          <td>26.906816</td>
          <td>0.204020</td>
          <td>26.193067</td>
          <td>0.098144</td>
          <td>26.238733</td>
          <td>0.165269</td>
          <td>25.950146</td>
          <td>0.239477</td>
          <td>25.270690</td>
          <td>0.291776</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.957658</td>
          <td>0.263665</td>
          <td>26.140968</td>
          <td>0.112093</td>
          <td>26.184920</td>
          <td>0.104051</td>
          <td>25.860150</td>
          <td>0.127631</td>
          <td>25.800649</td>
          <td>0.225098</td>
          <td>26.203036</td>
          <td>0.626609</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.159421</td>
          <td>0.295140</td>
          <td>26.563314</td>
          <td>0.151173</td>
          <td>26.473252</td>
          <td>0.124081</td>
          <td>26.400388</td>
          <td>0.187667</td>
          <td>25.666794</td>
          <td>0.187213</td>
          <td>25.368727</td>
          <td>0.312711</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
