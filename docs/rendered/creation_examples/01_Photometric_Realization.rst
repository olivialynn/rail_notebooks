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

    <pzflow.flow.Flow at 0x7f04a0627550>



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
    0      23.994413  0.005597  0.005530  
    1      25.391064  0.083607  0.043064  
    2      24.304707  0.124061  0.107705  
    3      25.291103  0.043839  0.026451  
    4      25.096743  0.011657  0.007273  
    ...          ...       ...       ...  
    99995  24.737946  0.104805  0.101733  
    99996  24.224169  0.129351  0.088056  
    99997  25.613836  0.134321  0.071428  
    99998  25.274899  0.049974  0.025649  
    99999  25.699642  0.062346  0.044024  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.779703</td>
          <td>0.175920</td>
          <td>25.956445</td>
          <td>0.075871</td>
          <td>25.244459</td>
          <td>0.065945</td>
          <td>24.679188</td>
          <td>0.076498</td>
          <td>24.038370</td>
          <td>0.097748</td>
          <td>0.005597</td>
          <td>0.005530</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.192780</td>
          <td>0.248455</td>
          <td>26.466566</td>
          <td>0.118705</td>
          <td>26.176565</td>
          <td>0.149078</td>
          <td>25.964578</td>
          <td>0.231462</td>
          <td>26.025445</td>
          <td>0.500514</td>
          <td>0.083607</td>
          <td>0.043064</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.726234</td>
          <td>0.380757</td>
          <td>30.994316</td>
          <td>2.318465</td>
          <td>25.992054</td>
          <td>0.127138</td>
          <td>24.996684</td>
          <td>0.101151</td>
          <td>24.125022</td>
          <td>0.105453</td>
          <td>0.124061</td>
          <td>0.107705</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.883344</td>
          <td>0.987321</td>
          <td>28.427814</td>
          <td>0.638926</td>
          <td>27.248560</td>
          <td>0.230983</td>
          <td>26.373800</td>
          <td>0.176420</td>
          <td>25.633031</td>
          <td>0.175248</td>
          <td>24.998026</td>
          <td>0.222550</td>
          <td>0.043839</td>
          <td>0.026451</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.560834</td>
          <td>0.395200</td>
          <td>26.023388</td>
          <td>0.091462</td>
          <td>25.888817</td>
          <td>0.071467</td>
          <td>25.670144</td>
          <td>0.096009</td>
          <td>25.401680</td>
          <td>0.143797</td>
          <td>25.565253</td>
          <td>0.352370</td>
          <td>0.011657</td>
          <td>0.007273</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.363359</td>
          <td>0.123060</td>
          <td>25.400937</td>
          <td>0.046360</td>
          <td>25.160258</td>
          <td>0.061201</td>
          <td>25.056930</td>
          <td>0.106625</td>
          <td>24.943040</td>
          <td>0.212580</td>
          <td>0.104805</td>
          <td>0.101733</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.695008</td>
          <td>0.437857</td>
          <td>27.059025</td>
          <td>0.222445</td>
          <td>26.030307</td>
          <td>0.080984</td>
          <td>25.059170</td>
          <td>0.055950</td>
          <td>24.964817</td>
          <td>0.098366</td>
          <td>24.492649</td>
          <td>0.145068</td>
          <td>0.129351</td>
          <td>0.088056</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.209419</td>
          <td>0.636592</td>
          <td>26.688269</td>
          <td>0.162756</td>
          <td>26.317636</td>
          <td>0.104244</td>
          <td>26.605561</td>
          <td>0.214429</td>
          <td>25.785131</td>
          <td>0.199273</td>
          <td>25.234924</td>
          <td>0.270479</td>
          <td>0.134321</td>
          <td>0.071428</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.816249</td>
          <td>0.217242</td>
          <td>26.027491</td>
          <td>0.091791</td>
          <td>26.246401</td>
          <td>0.097939</td>
          <td>26.043940</td>
          <td>0.132979</td>
          <td>25.551681</td>
          <td>0.163523</td>
          <td>24.960071</td>
          <td>0.215624</td>
          <td>0.049974</td>
          <td>0.025649</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>29.686595</td>
          <td>2.361452</td>
          <td>26.752634</td>
          <td>0.171923</td>
          <td>26.525351</td>
          <td>0.124923</td>
          <td>26.538392</td>
          <td>0.202708</td>
          <td>25.883382</td>
          <td>0.216355</td>
          <td>25.741191</td>
          <td>0.404020</td>
          <td>0.062346</td>
          <td>0.044024</td>
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
          <td>27.569913</td>
          <td>0.883527</td>
          <td>26.752184</td>
          <td>0.197207</td>
          <td>26.354965</td>
          <td>0.126410</td>
          <td>25.171550</td>
          <td>0.073283</td>
          <td>24.600463</td>
          <td>0.083947</td>
          <td>23.922073</td>
          <td>0.104290</td>
          <td>0.005597</td>
          <td>0.005530</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.252571</td>
          <td>0.301376</td>
          <td>26.412409</td>
          <td>0.134784</td>
          <td>26.502033</td>
          <td>0.234068</td>
          <td>26.567764</td>
          <td>0.438597</td>
          <td>25.566756</td>
          <td>0.414266</td>
          <td>0.083607</td>
          <td>0.043064</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.675703</td>
          <td>0.490812</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.472761</td>
          <td>0.334866</td>
          <td>25.886634</td>
          <td>0.143226</td>
          <td>25.009821</td>
          <td>0.125432</td>
          <td>24.251013</td>
          <td>0.145031</td>
          <td>0.124061</td>
          <td>0.107705</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.070988</td>
          <td>1.074144</td>
          <td>27.171063</td>
          <td>0.253191</td>
          <td>26.364668</td>
          <td>0.206673</td>
          <td>25.483402</td>
          <td>0.181150</td>
          <td>25.377352</td>
          <td>0.354352</td>
          <td>0.043839</td>
          <td>0.026451</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.036701</td>
          <td>0.290453</td>
          <td>25.986432</td>
          <td>0.102166</td>
          <td>25.893678</td>
          <td>0.084471</td>
          <td>25.855981</td>
          <td>0.133454</td>
          <td>25.763976</td>
          <td>0.228291</td>
          <td>25.271473</td>
          <td>0.324667</td>
          <td>0.011657</td>
          <td>0.007273</td>
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
          <td>27.900578</td>
          <td>1.099034</td>
          <td>26.327024</td>
          <td>0.141621</td>
          <td>25.552340</td>
          <td>0.064703</td>
          <td>25.230450</td>
          <td>0.080056</td>
          <td>24.852005</td>
          <td>0.108388</td>
          <td>24.684452</td>
          <td>0.207714</td>
          <td>0.104805</td>
          <td>0.101733</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.561167</td>
          <td>0.173729</td>
          <td>25.910225</td>
          <td>0.089173</td>
          <td>25.220704</td>
          <td>0.079763</td>
          <td>24.784612</td>
          <td>0.102674</td>
          <td>24.176796</td>
          <td>0.135490</td>
          <td>0.129351</td>
          <td>0.088056</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.809490</td>
          <td>1.043332</td>
          <td>26.687689</td>
          <td>0.192948</td>
          <td>26.330441</td>
          <td>0.128400</td>
          <td>26.413888</td>
          <td>0.222433</td>
          <td>25.979234</td>
          <td>0.281972</td>
          <td>26.180076</td>
          <td>0.660412</td>
          <td>0.134321</td>
          <td>0.071428</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.290084</td>
          <td>0.356505</td>
          <td>26.226457</td>
          <td>0.126459</td>
          <td>26.125549</td>
          <td>0.104060</td>
          <td>26.126481</td>
          <td>0.169167</td>
          <td>25.717486</td>
          <td>0.220684</td>
          <td>25.744509</td>
          <td>0.469945</td>
          <td>0.049974</td>
          <td>0.025649</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.241881</td>
          <td>0.344307</td>
          <td>26.865446</td>
          <td>0.218621</td>
          <td>26.524895</td>
          <td>0.147777</td>
          <td>25.859123</td>
          <td>0.135117</td>
          <td>25.725914</td>
          <td>0.223180</td>
          <td>25.241995</td>
          <td>0.319951</td>
          <td>0.062346</td>
          <td>0.044024</td>
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
          <td>26.692575</td>
          <td>0.437161</td>
          <td>26.821950</td>
          <td>0.182393</td>
          <td>26.067927</td>
          <td>0.083751</td>
          <td>25.308687</td>
          <td>0.069836</td>
          <td>24.666249</td>
          <td>0.075660</td>
          <td>23.914706</td>
          <td>0.087724</td>
          <td>0.005597</td>
          <td>0.005530</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.406799</td>
          <td>0.309174</td>
          <td>26.546302</td>
          <td>0.134378</td>
          <td>26.088178</td>
          <td>0.146271</td>
          <td>25.588270</td>
          <td>0.178016</td>
          <td>25.219574</td>
          <td>0.281748</td>
          <td>0.083607</td>
          <td>0.043064</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.020091</td>
          <td>1.748920</td>
          <td>28.366168</td>
          <td>0.626005</td>
          <td>26.245381</td>
          <td>0.184895</td>
          <td>25.158988</td>
          <td>0.135787</td>
          <td>24.222283</td>
          <td>0.134463</td>
          <td>0.124061</td>
          <td>0.107705</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.302169</td>
          <td>1.265669</td>
          <td>28.143534</td>
          <td>0.528201</td>
          <td>27.315349</td>
          <td>0.248106</td>
          <td>26.246440</td>
          <td>0.161119</td>
          <td>25.761454</td>
          <td>0.198626</td>
          <td>25.439547</td>
          <td>0.324261</td>
          <td>0.043839</td>
          <td>0.026451</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.359892</td>
          <td>0.338101</td>
          <td>26.122279</td>
          <td>0.099854</td>
          <td>26.032060</td>
          <td>0.081214</td>
          <td>25.627341</td>
          <td>0.092592</td>
          <td>25.468344</td>
          <td>0.152463</td>
          <td>25.227347</td>
          <td>0.269146</td>
          <td>0.011657</td>
          <td>0.007273</td>
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
          <td>26.313350</td>
          <td>0.353378</td>
          <td>26.392268</td>
          <td>0.140822</td>
          <td>25.408230</td>
          <td>0.053050</td>
          <td>25.117564</td>
          <td>0.067354</td>
          <td>24.772665</td>
          <td>0.094308</td>
          <td>24.919851</td>
          <td>0.236025</td>
          <td>0.104805</td>
          <td>0.101733</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.021336</td>
          <td>0.282786</td>
          <td>27.120551</td>
          <td>0.262961</td>
          <td>25.867781</td>
          <td>0.080825</td>
          <td>25.343440</td>
          <td>0.083437</td>
          <td>24.707705</td>
          <td>0.090335</td>
          <td>24.255971</td>
          <td>0.136442</td>
          <td>0.129351</td>
          <td>0.088056</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.134027</td>
          <td>1.215700</td>
          <td>26.345360</td>
          <td>0.135866</td>
          <td>26.281866</td>
          <td>0.115133</td>
          <td>26.728920</td>
          <td>0.269958</td>
          <td>26.395922</td>
          <td>0.369518</td>
          <td>25.562635</td>
          <td>0.396571</td>
          <td>0.134321</td>
          <td>0.071428</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.226569</td>
          <td>0.307877</td>
          <td>26.376687</td>
          <td>0.126738</td>
          <td>26.247592</td>
          <td>0.100102</td>
          <td>25.871367</td>
          <td>0.116988</td>
          <td>25.979816</td>
          <td>0.239033</td>
          <td>26.119501</td>
          <td>0.545844</td>
          <td>0.049974</td>
          <td>0.025649</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.509539</td>
          <td>0.388828</td>
          <td>27.067942</td>
          <td>0.231196</td>
          <td>26.783396</td>
          <td>0.161903</td>
          <td>26.547464</td>
          <td>0.212128</td>
          <td>25.790808</td>
          <td>0.207587</td>
          <td>25.827610</td>
          <td>0.446466</td>
          <td>0.062346</td>
          <td>0.044024</td>
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
