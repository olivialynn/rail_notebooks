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

    <pzflow.flow.Flow at 0x7f62fbfefcd0>



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
    0      23.994413  0.081430  0.067982  
    1      25.391064  0.092427  0.078128  
    2      24.304707  0.051808  0.030507  
    3      25.291103  0.140074  0.087237  
    4      25.096743  0.034676  0.029580  
    ...          ...       ...       ...  
    99995  24.737946  0.078316  0.057907  
    99996  24.224169  0.055004  0.033859  
    99997  25.613836  0.104591  0.057756  
    99998  25.274899  0.168312  0.114649  
    99999  25.699642  0.009670  0.009557  
    
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
          <td>29.624111</td>
          <td>2.306256</td>
          <td>26.589668</td>
          <td>0.149591</td>
          <td>25.971627</td>
          <td>0.076895</td>
          <td>25.165930</td>
          <td>0.061510</td>
          <td>24.778267</td>
          <td>0.083487</td>
          <td>24.088295</td>
          <td>0.102118</td>
          <td>0.081430</td>
          <td>0.067982</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.310128</td>
          <td>2.034972</td>
          <td>27.437170</td>
          <td>0.303037</td>
          <td>26.619924</td>
          <td>0.135579</td>
          <td>26.050131</td>
          <td>0.133692</td>
          <td>26.085265</td>
          <td>0.255669</td>
          <td>25.200819</td>
          <td>0.263057</td>
          <td>0.092427</td>
          <td>0.078128</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.765110</td>
          <td>0.392392</td>
          <td>27.789597</td>
          <td>0.357561</td>
          <td>25.833820</td>
          <td>0.110794</td>
          <td>25.240816</td>
          <td>0.125138</td>
          <td>24.366584</td>
          <td>0.130116</td>
          <td>0.051808</td>
          <td>0.030507</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.951796</td>
          <td>1.739953</td>
          <td>28.195803</td>
          <td>0.541827</td>
          <td>27.193445</td>
          <td>0.220647</td>
          <td>26.039203</td>
          <td>0.132435</td>
          <td>25.842546</td>
          <td>0.209100</td>
          <td>25.120768</td>
          <td>0.246341</td>
          <td>0.140074</td>
          <td>0.087237</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.844465</td>
          <td>0.222397</td>
          <td>26.113087</td>
          <td>0.098944</td>
          <td>25.923774</td>
          <td>0.073711</td>
          <td>25.701092</td>
          <td>0.098650</td>
          <td>25.247134</td>
          <td>0.125826</td>
          <td>25.231980</td>
          <td>0.269831</td>
          <td>0.034676</td>
          <td>0.029580</td>
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
          <td>26.962303</td>
          <td>0.533923</td>
          <td>26.574158</td>
          <td>0.147613</td>
          <td>25.380744</td>
          <td>0.045536</td>
          <td>25.081340</td>
          <td>0.057062</td>
          <td>24.843070</td>
          <td>0.088390</td>
          <td>24.872438</td>
          <td>0.200374</td>
          <td>0.078316</td>
          <td>0.057907</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.836069</td>
          <td>0.184519</td>
          <td>25.998859</td>
          <td>0.078767</td>
          <td>25.171994</td>
          <td>0.061841</td>
          <td>24.895357</td>
          <td>0.092549</td>
          <td>24.112708</td>
          <td>0.104323</td>
          <td>0.055004</td>
          <td>0.033859</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.917137</td>
          <td>1.007620</td>
          <td>27.294388</td>
          <td>0.269994</td>
          <td>26.259774</td>
          <td>0.099094</td>
          <td>26.482654</td>
          <td>0.193428</td>
          <td>26.898083</td>
          <td>0.483539</td>
          <td>25.583304</td>
          <td>0.357399</td>
          <td>0.104591</td>
          <td>0.057756</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.969080</td>
          <td>0.536557</td>
          <td>26.154876</td>
          <td>0.102628</td>
          <td>25.962322</td>
          <td>0.076266</td>
          <td>26.001599</td>
          <td>0.128194</td>
          <td>25.474724</td>
          <td>0.153107</td>
          <td>25.408909</td>
          <td>0.311274</td>
          <td>0.168312</td>
          <td>0.114649</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.321170</td>
          <td>0.687551</td>
          <td>26.513531</td>
          <td>0.140115</td>
          <td>26.545252</td>
          <td>0.127097</td>
          <td>26.557191</td>
          <td>0.205928</td>
          <td>26.003262</td>
          <td>0.238988</td>
          <td>25.562657</td>
          <td>0.351651</td>
          <td>0.009670</td>
          <td>0.009557</td>
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
          <td>26.723135</td>
          <td>0.195584</td>
          <td>26.198165</td>
          <td>0.112364</td>
          <td>25.160703</td>
          <td>0.074013</td>
          <td>24.726747</td>
          <td>0.095575</td>
          <td>23.952226</td>
          <td>0.109139</td>
          <td>0.081430</td>
          <td>0.067982</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.810603</td>
          <td>1.750562</td>
          <td>28.037782</td>
          <td>0.553491</td>
          <td>26.776324</td>
          <td>0.185684</td>
          <td>26.632398</td>
          <td>0.262975</td>
          <td>25.585601</td>
          <td>0.201249</td>
          <td>25.848576</td>
          <td>0.515965</td>
          <td>0.092427</td>
          <td>0.078128</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.454408</td>
          <td>2.278940</td>
          <td>29.373731</td>
          <td>1.275062</td>
          <td>27.176653</td>
          <td>0.254751</td>
          <td>26.256739</td>
          <td>0.189059</td>
          <td>24.958461</td>
          <td>0.115561</td>
          <td>24.186606</td>
          <td>0.132072</td>
          <td>0.051808</td>
          <td>0.030507</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.884150</td>
          <td>1.823858</td>
          <td>28.698350</td>
          <td>0.877796</td>
          <td>27.118374</td>
          <td>0.251584</td>
          <td>26.129393</td>
          <td>0.176257</td>
          <td>25.334310</td>
          <td>0.165809</td>
          <td>25.144690</td>
          <td>0.305580</td>
          <td>0.140074</td>
          <td>0.087237</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.500114</td>
          <td>0.418873</td>
          <td>26.215135</td>
          <td>0.125023</td>
          <td>26.076943</td>
          <td>0.099546</td>
          <td>25.626266</td>
          <td>0.109676</td>
          <td>25.293639</td>
          <td>0.153978</td>
          <td>24.976583</td>
          <td>0.256633</td>
          <td>0.034676</td>
          <td>0.029580</td>
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
          <td>26.824147</td>
          <td>0.537458</td>
          <td>26.394779</td>
          <td>0.147580</td>
          <td>25.513960</td>
          <td>0.061339</td>
          <td>25.100414</td>
          <td>0.069958</td>
          <td>24.916005</td>
          <td>0.112450</td>
          <td>24.396532</td>
          <td>0.159753</td>
          <td>0.078316</td>
          <td>0.057907</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.068616</td>
          <td>1.192596</td>
          <td>26.637402</td>
          <td>0.180091</td>
          <td>25.983382</td>
          <td>0.092025</td>
          <td>25.152642</td>
          <td>0.072593</td>
          <td>24.885608</td>
          <td>0.108551</td>
          <td>24.255287</td>
          <td>0.140269</td>
          <td>0.055004</td>
          <td>0.033859</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.402636</td>
          <td>0.393941</td>
          <td>27.078418</td>
          <td>0.263660</td>
          <td>26.147307</td>
          <td>0.108000</td>
          <td>26.302109</td>
          <td>0.199847</td>
          <td>25.765362</td>
          <td>0.233615</td>
          <td>25.231578</td>
          <td>0.321394</td>
          <td>0.104591</td>
          <td>0.057756</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.635118</td>
          <td>0.483083</td>
          <td>26.006817</td>
          <td>0.110298</td>
          <td>26.007569</td>
          <td>0.099644</td>
          <td>25.749807</td>
          <td>0.130078</td>
          <td>26.053259</td>
          <td>0.307062</td>
          <td>24.835669</td>
          <td>0.242544</td>
          <td>0.168312</td>
          <td>0.114649</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.980453</td>
          <td>0.277543</td>
          <td>26.865830</td>
          <td>0.216920</td>
          <td>26.761133</td>
          <td>0.179123</td>
          <td>26.433878</td>
          <td>0.218085</td>
          <td>25.615259</td>
          <td>0.201647</td>
          <td>25.495599</td>
          <td>0.387120</td>
          <td>0.009670</td>
          <td>0.009557</td>
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
          <td>27.042904</td>
          <td>0.589410</td>
          <td>27.108316</td>
          <td>0.245621</td>
          <td>25.929843</td>
          <td>0.079583</td>
          <td>25.180162</td>
          <td>0.067136</td>
          <td>24.611617</td>
          <td>0.077385</td>
          <td>23.957267</td>
          <td>0.097964</td>
          <td>0.081430</td>
          <td>0.067982</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.796007</td>
          <td>1.674386</td>
          <td>27.096643</td>
          <td>0.247224</td>
          <td>26.739216</td>
          <td>0.164046</td>
          <td>26.176494</td>
          <td>0.163424</td>
          <td>25.591613</td>
          <td>0.184570</td>
          <td>24.793253</td>
          <td>0.204996</td>
          <td>0.092427</td>
          <td>0.078128</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.906267</td>
          <td>0.399833</td>
          <td>25.911138</td>
          <td>0.121481</td>
          <td>25.035144</td>
          <td>0.107122</td>
          <td>24.187678</td>
          <td>0.114152</td>
          <td>0.051808</td>
          <td>0.030507</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.994406</td>
          <td>0.598330</td>
          <td>31.498413</td>
          <td>3.050868</td>
          <td>27.121332</td>
          <td>0.240059</td>
          <td>26.248335</td>
          <td>0.184945</td>
          <td>25.126722</td>
          <td>0.131780</td>
          <td>25.378915</td>
          <td>0.350486</td>
          <td>0.140074</td>
          <td>0.087237</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.326404</td>
          <td>0.331877</td>
          <td>25.973377</td>
          <td>0.088597</td>
          <td>25.956982</td>
          <td>0.076978</td>
          <td>25.651604</td>
          <td>0.095848</td>
          <td>25.624180</td>
          <td>0.176287</td>
          <td>25.000336</td>
          <td>0.226046</td>
          <td>0.034676</td>
          <td>0.029580</td>
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
          <td>27.569003</td>
          <td>0.836475</td>
          <td>26.583101</td>
          <td>0.156588</td>
          <td>25.375987</td>
          <td>0.048197</td>
          <td>25.029271</td>
          <td>0.058075</td>
          <td>24.800463</td>
          <td>0.090428</td>
          <td>24.549872</td>
          <td>0.161926</td>
          <td>0.078316</td>
          <td>0.057907</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.327672</td>
          <td>0.335062</td>
          <td>26.621293</td>
          <td>0.157315</td>
          <td>25.908676</td>
          <td>0.074770</td>
          <td>25.164496</td>
          <td>0.063243</td>
          <td>24.843714</td>
          <td>0.090899</td>
          <td>24.222849</td>
          <td>0.118134</td>
          <td>0.055004</td>
          <td>0.033859</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.145379</td>
          <td>0.639364</td>
          <td>26.613081</td>
          <td>0.164331</td>
          <td>26.392215</td>
          <td>0.121210</td>
          <td>26.264885</td>
          <td>0.175499</td>
          <td>26.310261</td>
          <td>0.331830</td>
          <td>25.491279</td>
          <td>0.360132</td>
          <td>0.104591</td>
          <td>0.057756</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.770985</td>
          <td>0.530710</td>
          <td>26.066241</td>
          <td>0.115162</td>
          <td>26.371965</td>
          <td>0.135490</td>
          <td>25.697497</td>
          <td>0.123064</td>
          <td>25.701344</td>
          <td>0.228332</td>
          <td>26.074994</td>
          <td>0.622918</td>
          <td>0.168312</td>
          <td>0.114649</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.397833</td>
          <td>0.348358</td>
          <td>26.802752</td>
          <td>0.179579</td>
          <td>26.276572</td>
          <td>0.100689</td>
          <td>26.369099</td>
          <td>0.175941</td>
          <td>25.824237</td>
          <td>0.206166</td>
          <td>25.905734</td>
          <td>0.458350</td>
          <td>0.009670</td>
          <td>0.009557</td>
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
