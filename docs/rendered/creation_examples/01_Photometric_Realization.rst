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

    <pzflow.flow.Flow at 0x7f474b93a050>



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
    0      23.994413  0.114753  0.112032  
    1      25.391064  0.006596  0.005069  
    2      24.304707  0.024102  0.016908  
    3      25.291103  0.096798  0.084331  
    4      25.096743  0.047354  0.027757  
    ...          ...       ...       ...  
    99995  24.737946  0.006335  0.004110  
    99996  24.224169  0.034871  0.027331  
    99997  25.613836  0.022226  0.013779  
    99998  25.274899  0.123748  0.095063  
    99999  25.699642  0.112517  0.092310  
    
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
          <td>26.922887</td>
          <td>0.518797</td>
          <td>26.649304</td>
          <td>0.157429</td>
          <td>26.058744</td>
          <td>0.083041</td>
          <td>25.237184</td>
          <td>0.065521</td>
          <td>24.674642</td>
          <td>0.076191</td>
          <td>23.916770</td>
          <td>0.087845</td>
          <td>0.114753</td>
          <td>0.112032</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.727102</td>
          <td>0.381013</td>
          <td>26.719193</td>
          <td>0.147682</td>
          <td>26.424765</td>
          <td>0.184204</td>
          <td>26.078818</td>
          <td>0.254321</td>
          <td>24.913884</td>
          <td>0.207460</td>
          <td>0.006596</td>
          <td>0.005069</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.380637</td>
          <td>0.715834</td>
          <td>28.902754</td>
          <td>0.875967</td>
          <td>28.301038</td>
          <td>0.526842</td>
          <td>26.173244</td>
          <td>0.148653</td>
          <td>25.055351</td>
          <td>0.106478</td>
          <td>24.367598</td>
          <td>0.130231</td>
          <td>0.024102</td>
          <td>0.016908</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.452123</td>
          <td>0.306694</td>
          <td>27.509749</td>
          <td>0.286084</td>
          <td>26.452062</td>
          <td>0.188502</td>
          <td>25.532390</td>
          <td>0.160852</td>
          <td>24.934481</td>
          <td>0.211065</td>
          <td>0.096798</td>
          <td>0.084331</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.175576</td>
          <td>0.291620</td>
          <td>26.157239</td>
          <td>0.102840</td>
          <td>26.063782</td>
          <td>0.083410</td>
          <td>25.672651</td>
          <td>0.096220</td>
          <td>25.574845</td>
          <td>0.166786</td>
          <td>24.884086</td>
          <td>0.202343</td>
          <td>0.047354</td>
          <td>0.027757</td>
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
          <td>27.185937</td>
          <td>0.626246</td>
          <td>26.431847</td>
          <td>0.130579</td>
          <td>25.467920</td>
          <td>0.049200</td>
          <td>24.986044</td>
          <td>0.052432</td>
          <td>24.775272</td>
          <td>0.083267</td>
          <td>24.483859</td>
          <td>0.143975</td>
          <td>0.006335</td>
          <td>0.004110</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.783370</td>
          <td>1.607634</td>
          <td>26.865768</td>
          <td>0.189204</td>
          <td>25.926118</td>
          <td>0.073864</td>
          <td>25.085151</td>
          <td>0.057255</td>
          <td>24.838768</td>
          <td>0.088056</td>
          <td>24.153152</td>
          <td>0.108077</td>
          <td>0.034871</td>
          <td>0.027331</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.767024</td>
          <td>0.462257</td>
          <td>27.006969</td>
          <td>0.213005</td>
          <td>26.543937</td>
          <td>0.126952</td>
          <td>26.120337</td>
          <td>0.142041</td>
          <td>26.055451</td>
          <td>0.249488</td>
          <td>25.585197</td>
          <td>0.357930</td>
          <td>0.022226</td>
          <td>0.013779</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.965819</td>
          <td>0.245846</td>
          <td>26.088337</td>
          <td>0.096822</td>
          <td>26.206201</td>
          <td>0.094545</td>
          <td>25.897041</td>
          <td>0.117068</td>
          <td>25.666221</td>
          <td>0.180251</td>
          <td>25.110475</td>
          <td>0.244262</td>
          <td>0.123748</td>
          <td>0.095063</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.736131</td>
          <td>0.451659</td>
          <td>26.670738</td>
          <td>0.160339</td>
          <td>26.604597</td>
          <td>0.133795</td>
          <td>26.195579</td>
          <td>0.151531</td>
          <td>26.011507</td>
          <td>0.240620</td>
          <td>25.761235</td>
          <td>0.410284</td>
          <td>0.112517</td>
          <td>0.092310</td>
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
          <td>26.638501</td>
          <td>0.476836</td>
          <td>26.686025</td>
          <td>0.193423</td>
          <td>25.979593</td>
          <td>0.094978</td>
          <td>25.126590</td>
          <td>0.073567</td>
          <td>24.854603</td>
          <td>0.109381</td>
          <td>24.111851</td>
          <td>0.128369</td>
          <td>0.114753</td>
          <td>0.112032</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.138995</td>
          <td>1.235922</td>
          <td>27.577781</td>
          <td>0.384975</td>
          <td>26.376295</td>
          <td>0.128769</td>
          <td>25.907008</td>
          <td>0.139434</td>
          <td>25.735958</td>
          <td>0.222997</td>
          <td>25.296772</td>
          <td>0.331195</td>
          <td>0.006596</td>
          <td>0.005069</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.203511</td>
          <td>0.561436</td>
          <td>26.014557</td>
          <td>0.153148</td>
          <td>24.908261</td>
          <td>0.110107</td>
          <td>24.225502</td>
          <td>0.135946</td>
          <td>0.024102</td>
          <td>0.016908</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.295163</td>
          <td>0.665036</td>
          <td>27.296693</td>
          <td>0.286481</td>
          <td>26.319391</td>
          <td>0.203512</td>
          <td>25.890321</td>
          <td>0.259817</td>
          <td>25.383664</td>
          <td>0.363635</td>
          <td>0.096798</td>
          <td>0.084331</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.929775</td>
          <td>0.267248</td>
          <td>26.103562</td>
          <td>0.113640</td>
          <td>25.923361</td>
          <td>0.087124</td>
          <td>25.654785</td>
          <td>0.112620</td>
          <td>25.481990</td>
          <td>0.181049</td>
          <td>25.758342</td>
          <td>0.474725</td>
          <td>0.047354</td>
          <td>0.027757</td>
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
          <td>26.288003</td>
          <td>0.354660</td>
          <td>26.070847</td>
          <td>0.109955</td>
          <td>25.440866</td>
          <td>0.056579</td>
          <td>25.002106</td>
          <td>0.063076</td>
          <td>24.792710</td>
          <td>0.099392</td>
          <td>24.920601</td>
          <td>0.244291</td>
          <td>0.006335</td>
          <td>0.004110</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.812192</td>
          <td>0.528605</td>
          <td>26.611921</td>
          <td>0.175675</td>
          <td>26.111963</td>
          <td>0.102625</td>
          <td>25.159645</td>
          <td>0.072761</td>
          <td>24.598794</td>
          <td>0.084096</td>
          <td>24.117712</td>
          <td>0.124072</td>
          <td>0.034871</td>
          <td>0.027331</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.473311</td>
          <td>0.155817</td>
          <td>26.219520</td>
          <td>0.112489</td>
          <td>26.352251</td>
          <td>0.203872</td>
          <td>26.121515</td>
          <td>0.305892</td>
          <td>25.538078</td>
          <td>0.400333</td>
          <td>0.022226</td>
          <td>0.013779</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.430320</td>
          <td>0.406817</td>
          <td>26.340280</td>
          <td>0.143840</td>
          <td>26.015018</td>
          <td>0.097756</td>
          <td>25.874875</td>
          <td>0.141183</td>
          <td>25.882602</td>
          <td>0.261180</td>
          <td>24.683147</td>
          <td>0.208433</td>
          <td>0.123748</td>
          <td>0.095063</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.896207</td>
          <td>0.572864</td>
          <td>26.572735</td>
          <td>0.174665</td>
          <td>26.715735</td>
          <td>0.178185</td>
          <td>26.237377</td>
          <td>0.191361</td>
          <td>25.999401</td>
          <td>0.285899</td>
          <td>25.789075</td>
          <td>0.498317</td>
          <td>0.112517</td>
          <td>0.092310</td>
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
          <td>27.667823</td>
          <td>0.932813</td>
          <td>26.483571</td>
          <td>0.155311</td>
          <td>26.144627</td>
          <td>0.103941</td>
          <td>25.265877</td>
          <td>0.078606</td>
          <td>24.642755</td>
          <td>0.086033</td>
          <td>24.088036</td>
          <td>0.118979</td>
          <td>0.114753</td>
          <td>0.112032</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.416404</td>
          <td>0.298133</td>
          <td>26.576835</td>
          <td>0.130682</td>
          <td>26.024553</td>
          <td>0.130831</td>
          <td>26.438824</td>
          <td>0.340054</td>
          <td>25.462428</td>
          <td>0.324996</td>
          <td>0.006596</td>
          <td>0.005069</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.856341</td>
          <td>0.378648</td>
          <td>26.399890</td>
          <td>0.181437</td>
          <td>24.933749</td>
          <td>0.096282</td>
          <td>24.299178</td>
          <td>0.123472</td>
          <td>0.024102</td>
          <td>0.016908</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.859415</td>
          <td>1.022100</td>
          <td>27.196183</td>
          <td>0.270484</td>
          <td>27.645657</td>
          <td>0.349326</td>
          <td>26.495565</td>
          <td>0.216135</td>
          <td>25.443585</td>
          <td>0.164385</td>
          <td>24.980097</td>
          <td>0.241859</td>
          <td>0.096798</td>
          <td>0.084331</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.362610</td>
          <td>0.342816</td>
          <td>26.159547</td>
          <td>0.104845</td>
          <td>25.943274</td>
          <td>0.076516</td>
          <td>25.550091</td>
          <td>0.088225</td>
          <td>25.417607</td>
          <td>0.148652</td>
          <td>24.956376</td>
          <td>0.219229</td>
          <td>0.047354</td>
          <td>0.027757</td>
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
          <td>26.736750</td>
          <td>0.451973</td>
          <td>26.470277</td>
          <td>0.135031</td>
          <td>25.452634</td>
          <td>0.048556</td>
          <td>24.990520</td>
          <td>0.052663</td>
          <td>25.005870</td>
          <td>0.102007</td>
          <td>24.536473</td>
          <td>0.150693</td>
          <td>0.006335</td>
          <td>0.004110</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.631103</td>
          <td>2.321571</td>
          <td>26.570646</td>
          <td>0.148821</td>
          <td>26.071326</td>
          <td>0.085083</td>
          <td>25.096746</td>
          <td>0.058661</td>
          <td>24.783858</td>
          <td>0.085013</td>
          <td>24.226443</td>
          <td>0.116781</td>
          <td>0.034871</td>
          <td>0.027331</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.591676</td>
          <td>0.824709</td>
          <td>27.322445</td>
          <td>0.277250</td>
          <td>26.429026</td>
          <td>0.115417</td>
          <td>26.303022</td>
          <td>0.166896</td>
          <td>25.784388</td>
          <td>0.200029</td>
          <td>25.448603</td>
          <td>0.322698</td>
          <td>0.022226</td>
          <td>0.013779</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.609236</td>
          <td>0.895766</td>
          <td>26.249889</td>
          <td>0.126140</td>
          <td>26.126494</td>
          <td>0.101502</td>
          <td>25.635509</td>
          <td>0.107862</td>
          <td>26.122368</td>
          <td>0.300123</td>
          <td>25.090023</td>
          <td>0.275144</td>
          <td>0.123748</td>
          <td>0.095063</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.601251</td>
          <td>0.440817</td>
          <td>26.819180</td>
          <td>0.202233</td>
          <td>26.587088</td>
          <td>0.149032</td>
          <td>26.450129</td>
          <td>0.213263</td>
          <td>25.559624</td>
          <td>0.185813</td>
          <td>25.158989</td>
          <td>0.286661</td>
          <td>0.112517</td>
          <td>0.092310</td>
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
